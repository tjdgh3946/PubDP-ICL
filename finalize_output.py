#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
finalize_output.py

Goal:
  - ALWAYS use an LLM to produce the final answer/summary.
  - Upstream step (DPM or KSA) only prepares CANDIDATES/HINTS.
  - Final decision/generation is done by the LLM in all cases.

Inputs:
  - output/<stem>_{public|private}.json          (raw generations with prompts/references)
  - semantic_group/[ood_]<stem>_<split>/<q>.npy  (embeddings per Q for DPM; optional)
  - (optional) semantic_group/...clusters_*.json (not required here)

Outputs:
  - final_outputs/<stem>_<split>_<method>_<task>.json

Conventions:
  - split ∈ {public, private, public+private}
  - <stem> = <dataset>{_subset?}_<model>_<ensemble>way-<ice_num>shot
"""

import os
import re
import json
import math
import argparse
from time import sleep
import sys

from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# ------- Metrics -------
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# ------- Original algorithms -------
# joint(): prefer semantic_group.jointEM, fall back to semantic_group.DPM
try:
    from semantic_group.jointEM import joint
except Exception:
    from semantic_group.DPM import joint  # type: ignore

from semantic_group.DPM import DPM  # DPM clustering


# =========================
# Utilities
# =========================
def require_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running.")


def stem_from_base(filename_base: str) -> str:
    return os.path.basename(filename_base)


def out_json_path(filename_base: str, split: str) -> str:
    return f"{filename_base}_{split}.json"


def sg_dir(stem: str, split: str, ood: bool) -> str:
    return os.path.join("semantic_group", f"{'ood_' if ood else ''}{stem}_{split}")


def sg_npy(stem: str, split: str, ood: bool, qidx: int) -> str:
    return os.path.join(sg_dir(stem, split, ood), f"{qidx}.npy")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_q(
    output_file: str,
) -> Tuple[Dict[int, List[dict]], Dict[int, Optional[str]]]:
    data = load_json(output_file)
    per_q: Dict[int, List[dict]] = {}
    refs: Dict[int, Optional[str]] = {}
    for row in data:
        qi = int(row["Qidx"])
        per_q.setdefault(qi, []).append(row)
        if qi not in refs:
            refs[qi] = row.get("reference")
    # stabilize by Eidx
    for qi in per_q:
        per_q[qi] = sorted(per_q[qi], key=lambda r: int(r.get("Eidx", 0)))
    return dict(sorted(per_q.items())), refs


def items_for_split(
    filename_base: str, split: str
) -> Tuple[Dict[int, List[str]], Dict[int, Optional[str]], Dict[int, str]]:
    """
    Return: texts per Q, references, and the first prompt per Q.
    """
    if split == "public+private":
        pub_rows, pub_refs = group_by_q(out_json_path(filename_base, "public"))
        pri_rows, pri_refs = group_by_q(out_json_path(filename_base, "private"))
        q_all = sorted(set(pub_rows.keys()) | set(pri_rows.keys()))
        items, refs, prompts = {}, {}, {}
        for qi in q_all:
            pub_items = [r["prediction"] for r in pub_rows.get(qi, [])]
            pri_items = [r["prediction"] for r in pri_rows.get(qi, [])]
            items[qi] = pub_items + pri_items
            refs[qi] = (
                pri_refs.get(qi) if pri_refs.get(qi) is not None else pub_refs.get(qi)
            )
            # Prefer public prompt for context
            if pub_rows.get(qi):
                prompts[qi] = pub_rows[qi][0]["prompt"]
            elif pri_rows.get(qi):
                prompts[qi] = pri_rows[qi][0]["prompt"]
            else:
                prompts[qi] = ""
        return items, refs, prompts
    else:
        rows, refs = group_by_q(out_json_path(filename_base, split))
        items = {qi: [r["prediction"] for r in rs] for qi, rs in rows.items()}
        prompts = {qi: rows[qi][0]["prompt"] if rows.get(qi) else "" for qi in rows}
        return items, refs, prompts


def public_count_per_q(filename_base: str) -> Dict[int, int]:
    pub_rows, _ = group_by_q(out_json_path(filename_base, "public"))
    return {qi: len(rs) for qi, rs in pub_rows.items()}


# =========================
# Prompt parsing & building
# =========================
def parse_qa_from_prompt(prompt: str) -> Tuple[str, str]:
    """
    Extract (public_ref, question) from QA-style prompt "Question:\n<ref>Question:\n<q>..."
    """
    if "Question:" in prompt:
        parts = prompt.split("Question:")
        if len(parts) >= 3:
            return parts[-2].strip(), parts[-1].strip()
        if len(parts) == 2:
            return "", parts[-1].strip()
    return "", prompt.strip()


def parse_sum_from_prompt(prompt: str) -> Tuple[str, str]:
    """
    Extract (public_ref, dialogue) from Summarization-style prompt "Dialogue:\n<ref>Dialogue:\n<d>..."
    """
    if "Dialogue:" in prompt:
        parts = prompt.split("Dialogue:")
        if len(parts) >= 3:
            return parts[-2].strip(), parts[-1].strip()
        if len(parts) == 2:
            return "", parts[-1].strip()
    return "", prompt.strip()


def build_public1shot_prompt(
    task: str,
    mode: str,
    public_example: str,
    query_text: str,
    candidates: List[str],
    *,
    ksa: bool = False,  # << set True when using KSA (joint) prompts
) -> str:
    """
    mode ∈ {"select","generate"}  (ignored when ksa=True)
    - For QA:    "Question:" ... (KSA: fixed instruction line)
    - For Summ.: "Dialogue:" ... (KSA: fixed instruction line)
    Candidate list is placed in [...] form.
    """
    if task == "summarization":
        public_ref, q_text = parse_sum_from_prompt(public_example)
        q_for_prompt = (query_text or q_text).strip()

        if ksa:
            # KSA instruction (summarization)
            # Summarize the above dialogue with the following word suggestions ranked by their frequency from high to low:
            head = "Dialogue:\n" + public_ref + "Dialogue:\n" + q_for_prompt + "\n"
            instr = "Summarize the above dialogue with the following word suggestions ranked by their frequency from high to low:\n"
            tail = ""  # no trailing "The summary is:" line for KSA style
        else:
            if mode == "select":
                q_mod = (
                    q_for_prompt
                    + " Pick the most accurate summary for the dialogue with the following summary suggestions:"
                )
            else:
                q_mod = (
                    q_for_prompt
                    + " Generate summary for the dialogue with the reference summary: "
                )
            head = "Dialogue:\n" + public_ref + "Dialogue:\n" + q_mod
            instr = ""
            tail = "\nThe summary is:"

    else:  # task == "qa"
        public_ref, q_text = parse_qa_from_prompt(public_example)
        q_for_prompt = (query_text or q_text).strip()

        if ksa:
            # KSA instruction (QA)
            # Answer the above question with the following word suggestions ranked by their frequency from high to low:
            head = "Question:\n" + public_ref + "Question:\n" + q_for_prompt + "\n"
            instr = "Answer the above question with the following word suggestions ranked by their frequency from high to low:\n"
            tail = ""  # no trailing "The answer is:" line for KSA style
        else:
            if mode == "select":
                q_mod = (
                    q_for_prompt
                    + "\nPick the most accurate answer for the question with the following answer candidates ranked by their frequency from high to low: "
                )
            else:
                q_mod = (
                    q_for_prompt
                    + " Generate answer for the question with the reference answer: "
                )
            head = "Question:\n" + public_ref + "Question:\n" + q_mod
            instr = ""
            tail = "\nThe answer is:"

    list_block = "[" + ",\n".join(candidates) + "]"
    return head + instr + list_block + tail


def _is_completion_model(model: str) -> bool:
    """
    Return True for legacy text-completion models (e.g., davinci-002, text-davinci-003),
    False for chat-completion style models (e.g., gpt-3.5-turbo, gpt-4*).
    """
    m = model.lower()
    return ("davinci" in m and not m.startswith("gpt")) or m.startswith("text-")


def llm_complete(
    prompt: str, model: str, max_tokens: int, temperature: float, stop=None
) -> str:
    """
    Legacy completion endpoint (for davinci-family).
    """
    import openai

    resp = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )
    txt = resp["choices"][0].get("text", "")
    return (txt or "").strip()


def llm_chat(
    prompt: str, model: str, max_tokens: int, temperature: float, stop=None
) -> str:
    """
    Chat-completion endpoint (for gpt-3.5/4-family).
    """
    import openai

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )
    return resp["choices"][0]["message"]["content"].strip()


def llm_generate(
    prompt: str, model: str, max_tokens: int, temperature: float, stop=None
) -> str:
    """
    Unified wrapper:
      - davinci/text-* -> Completion
      - gpt-*          -> ChatCompletion
    """
    if _is_completion_model(model):
        return llm_complete(prompt, model, max_tokens, temperature, stop=stop)
    return llm_chat(prompt, model, max_tokens, temperature, stop=stop)


# =========================
# DPM candidate selection
# =========================
def nearest_idx(center: np.ndarray, pool: np.ndarray) -> int:
    d = np.linalg.norm(pool - center, axis=1)
    return int(np.argmin(d))


def dpm_pick_candidates(
    embeddings: np.ndarray,
    texts: List[str],
    pool_slice: slice,
    eps: float,
    eps_em: float,
    eps_gm: float,
    levels: int,
    k: int,
) -> List[str]:
    if embeddings is None or embeddings.size == 0 or len(texts) == 0:
        return []

    lb, ub = float(np.min(embeddings)), float(np.max(embeddings))
    n = embeddings.shape[0]
    delta = 1.0 / (math.sqrt(n) * n) if n > 0 else 1e-6

    if math.isinf(eps):
        eps_exp = 1e6
        eps_avg = 1e6
    else:
        eps_exp = eps_em / max(eps, 1e-6)
        eps_avg = eps_gm / max(eps, 1e-6)

    dpm = DPM(
        data=embeddings,
        bounds=(lb, ub),
        epsilon=eps,
        epsilon_scale_exp=eps_exp,
        epsilon_scale_average=eps_avg,
        num_splits_levels=levels,
        debug=False,
        delta=delta,
    )
    centers, clusters = dpm.perform_clustering()

    order = sorted(range(len(clusters)), key=lambda i: len(clusters[i]), reverse=True)
    centers = [centers[i] for i in order]

    pool = embeddings[pool_slice]
    offset = pool_slice.start or 0

    chosen: List[str] = []
    used_idx: set = set()
    for c in centers:
        if pool.shape[0] == 0:
            break
        j = nearest_idx(c, pool)
        gi = offset + j
        if gi in used_idx:
            continue
        used_idx.add(gi)
        chosen.append(texts[gi])
        if len(chosen) >= k:
            break
    return chosen


# =========================
# KSA token selection (joint)
# =========================
def ksa_select_tokens(all_texts: List[str], k: int, eps: float) -> List[str]:
    """
    1-gram sentence-level counting, exponential mechanism (joint) or top-k if eps=inf.
    """
    from nltk.util import ngrams
    import string
    from nltk.corpus import stopwords

    stopword_set = set(stopwords.words("english"))

    counts: Dict[Tuple[str], int] = {}
    for t in all_texts:
        toks = nltk.word_tokenize(t)
        onegrams = set(ngrams(toks, 1))
        for tok in onegrams:
            counts[tok] = counts.get(tok, 0) + 1

    cleaned = []
    for tok, c in counts.items():
        w = tok[0]
        if all(ch in string.punctuation for ch in w):
            continue
        if w in stopword_set:
            continue
        cleaned.append((tok, c))

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: x[1], reverse=True)
    arr = np.array([c for (_, c) in cleaned], dtype=np.float32)
    k_eff = min(k, len(arr))

    if math.isinf(eps):
        idx = np.argpartition(-arr, range(k_eff))[:k_eff]
        idx = idx[np.argsort(-arr[idx])]
    else:
        idx = joint(arr, k_eff, epsilon=eps, neighbor_type=1)

    return [cleaned[i][0][0] for i in idx]


# =========================
# Metrics
# =========================
def scores_qa(
    preds: List[str], refs: List[Optional[str]]
) -> Dict[str, Dict[str, float]]:
    rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    b, m, r1 = [], [], []
    for p, r in zip(preds, refs):
        if not r:
            continue
        pt, rt = word_tokenize(p), word_tokenize(r)
        b.append(
            sentence_bleu([rt], pt, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)
        )
        m.append(meteor_score([rt], pt))
        r1.append(rouge.score(r, p)["rouge1"].fmeasure)

    def ms(vals):
        if not vals:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(np.mean(vals) * 100), "std": float(np.std(vals) * 100)}

    return {"bleu": ms(b), "meteor": ms(m), "rouge1": ms(r1)}


def scores_sum(
    preds: List[str], refs: List[Optional[str]]
) -> Dict[str, Dict[str, float]]:
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for p, r in zip(preds, refs):
        if not r:
            continue
        s = rouge.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)

    def ms(vals):
        if not vals:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(np.mean(vals) * 100), "std": float(np.std(vals) * 100)}

    return {"rouge1": ms(r1), "rouge2": ms(r2), "rougeL": ms(rL)}


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="Finalize outputs: DPM/KSA prepares candidates → LLM ALWAYS decides."
    )
    ap.add_argument(
        "--filename_base",
        required=True,
        help="e.g., output/medical-question-answering-datasets_gpt-3.5-turbo_100way-4shot",
    )
    ap.add_argument(
        "--split", default="public", choices=["public", "private", "public+private"]
    )
    ap.add_argument("--ood", action="store_true")

    # Task (affects prompt style & metrics)
    ap.add_argument("--task", default="qa", choices=["qa", "summarization"])

    # Method to build candidates/hints (LLM is always used at the end)
    ap.add_argument("--method", default="dpm", choices=["dpm", "ksa"])

    # DPM knobs
    ap.add_argument("--dpm_eps", type=float, default=1.0)
    ap.add_argument("--dpm_eps_em", type=float, default=0.12)
    ap.add_argument("--dpm_eps_gm", type=float, default=7.51)
    ap.add_argument("--dpm_levels", type=int, default=4)
    ap.add_argument(
        "--dpm_pool",
        type=str,
        default="auto",
        choices=["auto", "public", "private", "all"],
        help="Which subset to draw DPM representatives from. For public+private, 'auto' uses public.",
    )

    # KSA knobs
    ap.add_argument("--ksa_eps", type=float, default=float("inf"))
    ap.add_argument("--ksa_k", type=int, default=40)
    ap.add_argument(
        "--ksa_source",
        type=str,
        default="public+private",
        choices=["public", "private", "public+private", "split"],
        help="Where to collect tokens from when building KSA hints.",
    )

    # Candidate cap for LLM select
    ap.add_argument("--topk", type=int, default=3)

    # LLM
    ap.add_argument("--model", type=str, default="gpt-3.5-turbo")
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.02)

    # Metrics
    ap.add_argument("--metrics", default="auto", choices=["auto", "qa", "sum", "none"])

    # Output dir
    ap.add_argument("--save_dir", type=str, default="final_outputs")

    args = ap.parse_args()
    require_api_key()

    stem = stem_from_base(args.filename_base)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load items/prompts
    texts_per_q, refs, prompts = items_for_split(args.filename_base, args.split)
    pub_len = (
        public_count_per_q(args.filename_base) if args.split == "public+private" else {}
    )

    final_records = []
    preds, golds = [], []

    for qi in tqdm(
        sorted(texts_per_q.keys()), desc=f"Finalize [{args.method}] {args.split}"
    ):
        texts = texts_per_q[qi]
        ref = refs.get(qi)
        public_example = prompts.get(qi, "")  # we use this as the context holder
        # Default: try to extract a 'query' snippet from that public prompt
        query_text = ""
        if args.task == "summarization":
            _, query_text = parse_sum_from_prompt(public_example)
            # Optional: strip instruction phrase if present
            query_text = query_text.replace("Summarize the above dialogue:", "").strip()
        else:
            _, query_text = parse_qa_from_prompt(public_example)

        # -------------------------
        # Build candidates/hints
        # -------------------------
        candidates: List[str] = []

        if args.method == "dpm":
            # embeddings
            npy = sg_npy(stem, args.split, args.ood, qi)
            if not os.path.exists(npy):
                # Fallback: frequency-unique topK
                uniq = []
                seen = set()
                for t in texts:
                    if t not in seen:
                        seen.add(t)
                        uniq.append(t)
                candidates = uniq[: args.topk]
            else:
                emb = np.load(npy)
                # pool slice
                if args.dpm_pool == "auto":
                    if args.split == "public":
                        pool_slice = slice(0, len(texts))
                    elif args.split == "private":
                        pool_slice = slice(0, len(texts))
                    else:  # public+private
                        pool_slice = slice(0, pub_len.get(qi, 0))
                elif args.dpm_pool == "public":
                    pool_slice = slice(
                        0,
                        (
                            pub_len.get(qi, 0)
                            if args.split == "public+private"
                            else len(texts) if args.split == "public" else 0
                        ),
                    )
                elif args.dpm_pool == "private":
                    if args.split == "public+private":
                        start = pub_len.get(qi, 0)
                        pool_slice = slice(start, len(texts))
                    else:
                        pool_slice = slice(0, len(texts))
                else:  # all
                    pool_slice = slice(0, len(texts))

                candidates = dpm_pick_candidates(
                    embeddings=emb,
                    texts=texts,
                    pool_slice=pool_slice,
                    eps=args.dpm_eps,
                    eps_em=args.dpm_eps_em,
                    eps_gm=args.dpm_eps_gm,
                    levels=args.dpm_levels,
                    k=args.topk,
                )
                if not candidates:
                    # last fallback
                    candidates = texts[: args.topk]

        elif args.method == "ksa":
            # choose sources
            if args.ksa_source == "split":
                src_splits = [args.split]
            elif args.ksa_source == "public+private":
                src_splits = ["public", "private"]
            else:
                src_splits = [args.ksa_source]

            all_src_texts: List[str] = []
            for sp in src_splits:
                it, _, _ = items_for_split(args.filename_base, sp)
                if qi in it:
                    all_src_texts.extend(it[qi])

            token_hints = ksa_select_tokens(all_src_texts, args.ksa_k, args.ksa_eps)
            # Represent tokens as candidates for 1-shot prompt
            candidates = token_hints[: args.topk] if token_hints else []
        else:
            raise ValueError("Unknown method")

        # -------------------------
        # LLM ALWAYS DECIDES
        # -------------------------
        # select vs generate mode: if only one candidate/hint, use 'generate', else 'select'
        mode = "generate" if len(candidates) <= 1 else "select"
        prompt = build_public1shot_prompt(
            args.task,
            mode,
            public_example,
            query_text,
            candidates if candidates else [""],
            ksa=True if args.method == "ksa" else False,
        )

        # unified LLM call (supports both chat-* and davinci/text-* models)
        for attempt in range(3):
            try:
                pred = llm_generate(
                    prompt=prompt,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stop=None,
                )
                break
            except Exception as e:

                if attempt == 2:
                    sys.stderr.write(f"[LLM ERROR] {e}\n")
                    # fail-safe fallback: first candidate text or empty
                    pred = texts[0] if texts else ""
                else:
                    sleep(1.0)

        final_records.append(
            {
                "Qidx": int(qi),
                "prediction": pred,
                "reference": ref,
                "method": f"{args.method}+llm",  # emphasize LLM decision always used
                "split": args.split,
                "task": args.task,
                "ood": bool(args.ood),
                "num_candidates": int(len(candidates)),
            }
        )
        preds.append(pred)
        golds.append(ref)

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"{stem}_{args.split}_{args.method}+llm_{args.task}"
    out_path = os.path.join(args.save_dir, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {out_path}")

    # Metrics
    mode = args.metrics
    if mode == "auto":
        mode = "sum" if args.task == "summarization" else "qa"

    if mode == "qa":
        s = scores_qa(preds, golds)
        print("[SCORES][QA] mean±std (%):")
        for k, v in s.items():
            print(f"  {k.upper():7s}: {v['mean']:.2f} ± {v['std']:.2f}")
    elif mode == "sum":
        s = scores_sum(preds, golds)
        print("[SCORES][SUM] mean±std (%):")
        for k, v in s.items():
            print(f"  {k.upper():7s}: {v['mean']:.2f} ± {v['std']:.2f}")
    elif mode == "none":
        pass


if __name__ == "__main__":
    main()
