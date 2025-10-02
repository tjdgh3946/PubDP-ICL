#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end pipeline that merges (1) generation (from run.py) and
(2) semantic embedding / clustering (from semantic_cluster.py).

Stages:
  A. Data load & split (including ID / OOD)
  B. Retrieval & prompt ensemble generation (relies on gen_util.UWORetriever)
  C. LLM generation (relies on gen_util.complete)
  D. Save ensemble outputs as JSON (private/public)
  E. Build per-question ensembles from JSONs
  F. Compute embeddings via OpenAI Embeddings API (async, aiohttp)
  G. (Optional) similarity-based clustering & representatives export

Notes:
  - Requires OPENAI_API_KEY env var (do NOT hardcode in code).
  - Keeps compatibility with your existing gen_util (UWORetriever, complete).
  - Fixes several bugs from the original snippets (key mismatches, arg handling, etc.).
"""

import os
import sys
import json
import time
import math
import random
import argparse
import asyncio
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm

# ----------------------------
# External deps expected:
#   - gen_util.py that defines:
#       * UWORetriever(train_dataset, test_dataset, template)
#         .retrieve(ice_num, ensemble, ds_size)
#         .prompt_generate(i, ice_num, ensemble, output_field) -> List[str]
#       * complete(prompt, max_token, temp, model_name, instruction=None) -> str (awaitable)
#   - openicl PromptTemplate class
# ----------------------------
from openicl import PromptTemplate

# from openicl import TopkRetriever, ZeroRetriever, RandomRetriever, PateRetriever, BM25Retriever  # not required directly

# ==== Reproducibility ====
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# ==== Prompt templates, instructions, output fields ====
TEMPLATE_MAP: Dict[str, Tuple[str, Dict[str, str]]] = {
    # key는 "dataset_name" 혹은 "owner/dataset" 모두 지원하도록 아래에서 normalize 처리
    "samsum": (
        '</E>Dialogue:"\n </dialogue>" \nThe summary is: </summary>',
        {"dialogue": "</dialogue>", "summary": "</summary>"},
    ),
    "virattt/financial-qa-10K": (
        '</E>Context: "\n </context>"\n\nQuestion: </question>\nAnswer: </answer>',
        {"context": "</context>", "question": "</question>", "answer": "</answer>"},
    ),
    "Malikeh1375/medical-question-answering-datasets": (
        "</E>Question: </question>\n\nAnswer: </answer>\n",
        {"input": "</question>", "output": "</answer>"},
    ),
}
INSTRUCTION_MAP: Dict[str, Optional[str]] = {
    "samsum": "Summarize the following dialogue:\n\n",
    "virattt/financial-qa-10K": None,
    "Malikeh1375/medical-question-answering-datasets": (
        "You are a doctor, please answer the medical questions based on the patient's description.\n"
    ),
}
OUTPUT_FIELD_MAP: Dict[str, str] = {
    "samsum": "summary",
    "virattt/financial-qa-10K": "answer",
    "Malikeh1375/medical-question-answering-datasets": "output",
}


TEMPLATE_MAP["knkarthick/samsum"] = TEMPLATE_MAP["samsum"]
INSTRUCTION_MAP["knkarthick/samsum"] = INSTRUCTION_MAP["samsum"]
OUTPUT_FIELD_MAP["knkarthick/samsum"] = OUTPUT_FIELD_MAP["samsum"]


# ---------- Utilities ----------
def ensure_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. ")


def dataset_key_variants(name: str) -> List[str]:
    """Return possible keys for name, handling 'owner/dataset' vs 'dataset'."""
    if "/" in name:
        owner, short = name.split("/", 1)
        return [name, short]
    return [name]


def pick_from_map(name: str, mapping: Dict[str, any], desc: str):
    """Pick a value from a dict mapping by trying 'owner/dataset' then 'dataset' key."""
    for k in dataset_key_variants(name):
        if k in mapping:
            return mapping[k]
    raise KeyError(
        f"{desc} not found for dataset '{name}'. Available keys: {list(mapping.keys())}"
    )


def train_public_private_test_split(
    hf_dataset, data_name: str, ood: bool = False, ood_dataset=None, ds_size: int = 100
):
    """
    Returns:
      - private_dataset (or None in OOD mode)
      - public_dataset
      - testset
    Rules:
      * in-distribution: split train->(private/public), sample test (100)
      * OOD: public from 'dataset' train; test from 'ood_dataset' (sample 100)
      * For Malikeh1375/... follow original slicing logic
    """
    short = dataset_key_variants(data_name)[-1]

    def _sample_test(dset, field_for_len=None, start_end=None):
        if "test" in dset:
            test = dset["test"]
        else:
            test = dset["train"].train_test_split(test_size=100, seed=0)["test"]
        # Shuffle & cap to ds_size or 100 whichever smaller
        k = min(len(test), ds_size, 100)
        return test.shuffle(seed=0).select(range(k))

    if not ood:
        # In-distribution
        if (
            short == "medical-question-answering-datasets"
            or "Malikeh1375/medical-question-answering-datasets" in data_name
        ):
            # Follow your original sorted slice logic
            base = hf_dataset["train"]
            dataset_sorted = sorted(base, key=lambda x: len(x["output"]))
            subset = Dataset.from_list(dataset_sorted[3000:7000])
            split = subset.train_test_split(test_size=100, seed=0)
            train_set, test_set = split["train"], split["test"]
        else:
            if "test" in hf_dataset:
                train_set, test_set = hf_dataset["train"], _sample_test(hf_dataset)
            else:
                split = hf_dataset["train"].train_test_split(test_size=100, seed=0)
                train_set, test_set = split["train"], split["test"]

        # Split train into private/public (2:1)
        split_dataset = train_set.train_test_split(test_size=1 / 3, seed=0)
        private_dataset, public_dataset = split_dataset["train"], split_dataset["test"]
        # cap test set
        test_set = test_set.shuffle(seed=0).select(range(min(len(test_set), ds_size)))
        return private_dataset, public_dataset, test_set
    else:
        if ood_dataset is None:
            raise ValueError(
                "--ood 사용 시 --ood_subset(또는 별도의 OOD 데이터 로드)이 필요합니다."
            )
        # public from in-dist dataset
        if (
            short == "medical-question-answering-datasets"
            or "Malikeh1375/medical-question-answering-datasets" in data_name
        ):
            base_pub = hf_dataset["train"]
            dataset_sorted = sorted(base_pub, key=lambda x: len(x["output"]))
            public_dataset = Dataset.from_list(dataset_sorted[3000:7000])
        else:
            public_dataset = hf_dataset["train"]
        # test from OOD dataset
        if "test" in ood_dataset:
            test_candidate = ood_dataset["test"]
        else:
            test_candidate = ood_dataset["train"]
        test_set = test_candidate.shuffle(seed=0).select(
            range(min(len(test_candidate), ds_size, 100))
        )
        return None, public_dataset, test_set


# ---------- Stage A+B+C: Generation ----------
async def run_generation_stage(args) -> Tuple[str, List[str]]:
    """
    Runs retrieval + generation and dumps JSON per split (private/public).
    Returns:
      filename_base (without _{split}.json),
      splits_written (list like ["private","public"] or ["public"])
    """
    ensure_api_key()

    # Template & fields for this dataset
    template_str, template_vars = pick_from_map(
        args.data_name, TEMPLATE_MAP, "template"
    )
    output_field = pick_from_map(args.data_name, OUTPUT_FIELD_MAP, "output field")
    instruction = pick_from_map(args.data_name, INSTRUCTION_MAP, "instruction")

    template = PromptTemplate(template_str, template_vars, ice_token="</E>")

    # Load datasets
    if args.subset:
        dataset = load_dataset(args.data_name, args.subset, cache_dir="./data")
    else:
        dataset = load_dataset(args.data_name, cache_dir="./data")

    if args.ood:
        if not args.ood_subset:
            raise ValueError(
                "--ood 사용 시 --ood_subset을 지정하세요 (예: 다른 서브셋/데이터)."
            )
        ood_dataset = load_dataset(args.data_name, args.ood_subset, cache_dir="./data")
    else:
        ood_dataset = None

    private_dataset, public_dataset, testset = train_public_private_test_split(
        dataset,
        args.data_name,
        ood=args.ood,
        ood_dataset=ood_dataset,
        ds_size=args.ds_size,
    )

    # Prepare retrievers (requires gen_util.UWORetriever)
    from gen_util import UWORetriever, complete  # noqa: E402

    retrievers = {}
    splits: List[str] = []
    if private_dataset is not None:
        retrievers["private"] = UWORetriever(private_dataset, testset, template)
        splits.append("private")
    retrievers["public"] = UWORetriever(public_dataset, testset, template)
    if "public" not in splits:
        splits.append("public")

    # Trigger retrieval indexes (not strictly used below unless UWORetriever caches)
    for lbl, r in retrievers.items():
        _ = r.retrieve(args.ice_num, args.ensemble, args.ds_size)

    # Filename base
    def _base_name() -> str:
        model_tag = args.model_name
        en_tag = f"{args.ensemble}way-{args.ice_num}shot"
        if args.ood:
            subset_tag = args.ood_subset
            data_tag = (
                args.data_name.split("/")[-1]
                if "/" in args.data_name
                else args.data_name
            )
            return f"output/ood_{data_tag}_{subset_tag}_{model_tag}_{en_tag}"
        else:
            if args.subset:
                data_tag = (
                    f"{args.data_name.split('/')[-1]}_{args.subset}"
                    if "/" in args.data_name
                    else f"{args.data_name}_{args.subset}"
                )
            else:
                data_tag = (
                    args.data_name.split("/")[-1]
                    if "/" in args.data_name
                    else args.data_name
                )
            return f"output/{data_tag}_{model_tag}_{en_tag}"

    filename_base = _base_name()
    os.makedirs(os.path.dirname(filename_base), exist_ok=True)
    print(f"[GEN] Writing predictions under: {filename_base}_*.json")

    # Generate
    for split in splits:
        print("#" * 12 + f" Generating for [{split}] " + "#" * 12)
        r = retrievers[split]
        predictions: List[Dict] = []

        N = min(args.ds_size, len(testset))
        for i in tqdm(range(N), desc=f"{split}/examples"):
            prompt_ensemble = r.prompt_generate(
                i, args.ice_num, args.ensemble, output_field=output_field
            )
            reference = testset[i][output_field]
            # Generate each ensemble member
            for j, prompt in enumerate(prompt_ensemble):
                out = await complete(
                    prompt,
                    args.max_token,
                    args.temp,
                    args.model_name,
                    instruction=instruction,
                )
                predictions.append(
                    {
                        "Qidx": int(i),
                        "Eidx": int(j),
                        "prompt": prompt,
                        "prediction": out,
                        "reference": reference,
                    }
                )

        out_path = f"{filename_base}_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"[GEN] Saved: {out_path}")

    return filename_base, splits


# ---------- Stage E: Build ensembles from saved JSON ----------
def load_ensembles(json_path: str, tag: str, query_type: str, gamma: float = 1.0):
    """
    Read {filename_base}_{public|private}.json and bundle per Qidx.
    Returns:
      ensembles: Dict[int, List[[text, tag]]]
      queries:   List[str]  (same order as Qidx)
      references:List[str]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ensembles: Dict[int, List[List[str]]] = {}
    queries: List[str] = []
    references: List[str] = []

    # We will track first-seen Qidx order
    seen_q_order: List[int] = []

    for i, row in enumerate(data):
        qidx = int(row["Qidx"])
        eidx = int(row["Eidx"])
        prompt = row["prompt"]
        pred = row["prediction"]
        ref = row.get("reference", None)

        item = [pred, tag]

        if qidx not in ensembles:
            ensembles[qidx] = [item]
            seen_q_order.append(qidx)
            # extract query by query_type
            q = ""
            if query_type.upper() == "QA":
                # Expect "... Question: ...\nAnswer: ..." format
                if "Question:" in prompt:
                    q = prompt.split("Question:", 1)[-1].split("\n")[0].strip()
            elif query_type.lower() in ("summary", "summarization"):
                # Expect 'Dialogue:' presence
                if "Dialogue:" in prompt:
                    # robust slicing: between Dialogue:" and The summary is:
                    p_after = prompt.split("Dialogue:", 1)[-1]
                    if "The summary is" in p_after:
                        q = p_after.split("The summary is", 1)[0].strip().strip('"')
                    else:
                        q = p_after.strip().strip('"')
            else:
                # fallback: full prompt
                q = prompt
            queries.append(q)
            references.append(ref)
        else:
            ensembles[qidx].append(item)

        # subsample ensembles if gamma < 1.0
        if eidx % 100 == 99 and gamma < 1.0:
            ensembles[qidx] = random.sample(
                ensembles[qidx], max(1, int(len(ensembles[qidx]) * gamma))
            )

    # Normalize by Q order
    ordered = {qi: ensembles[qi] for qi in seen_q_order}

    return ordered, queries, references


# ---------- Stage F: Embeddings (async, aiohttp) ----------
import aiohttp


async def _embed_batch(
    session: aiohttp.ClientSession,
    inputs: List[str],
    model: str,
    semaphore: asyncio.Semaphore,
):
    """
    Send a single embeddings request (the API supports batching multiple inputs).
    """
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {"input": inputs, "model": model}

    async with semaphore:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"Embeddings API error {resp.status}: {txt}")
            j = await resp.json()
            # Return list of vectors (in same order as inputs)
            return [d["embedding"] for d in j["data"]]


async def compute_and_export_embeddings_per_q(
    filename_base: str,
    splits: List[str],
    query_type: str,
    embedding_model: str,
    aggregate: bool,
    gamma: float,
    export_embedding: bool,
    similarity_threshold: float,
    do_cluster: bool,
    ood: bool,
):
    """
    Export embeddings (and optional clusters) per split:
      - If *_public.json exists -> export under semantic_group/<stem>_public/
      - If *_private.json exists -> export under semantic_group/<stem>_private/
      - If both exist and `aggregate` -> also export merged under .../<stem>_public+private/
    """
    ensure_api_key()

    # --------------------------
    # 1) Load ensembles per split
    # --------------------------
    have_private = os.path.exists(f"{filename_base}_private.json")
    have_public = os.path.exists(f"{filename_base}_public.json")
    if not have_private and not have_public:
        raise FileNotFoundError(
            f"No ensemble JSONs found around {filename_base}_*.json"
        )

    base_ensembles = {}  # {"public": {qidx: [[text, tag], ...]}, "private": {...}}
    base_queries = {}  # {"public": [q0, q1, ...], "private": [...]}
    base_refs = {}  # {"public": [r0, r1, ...], "private": [...]}

    if have_private:
        pri, q_pri, r_pri = load_ensembles(
            f"{filename_base}_private.json", "private", query_type, gamma
        )
        base_ensembles["private"] = pri
        base_queries["private"] = q_pri
        base_refs["private"] = r_pri

    if have_public:
        pub, q_pub, r_pub = load_ensembles(
            f"{filename_base}_public.json", "public", query_type, gamma
        )
        base_ensembles["public"] = pub
        base_queries["public"] = q_pub
        base_refs["public"] = r_pub

    # helper: union of qidx keys (keep sorted for stability)
    def all_qindices(dicts):
        keyset = set()
        for d in dicts:
            keyset.update(d.keys())
        return sorted(keyset)

    # --------------------------
    # 2) Build work list: which splits to export
    #    - always export each existing split separately
    #    - if aggregate and both exist -> add merged "public+private"
    # --------------------------
    work_items = []  # list of (split_tag, merged_dict, queries, refs)

    if "public" in base_ensembles:
        work_items.append(
            (
                "public",
                base_ensembles["public"],
                base_queries["public"],
                base_refs["public"],
            )
        )

    if "private" in base_ensembles:
        work_items.append(
            (
                "private",
                base_ensembles["private"],
                base_queries["private"],
                base_refs["private"],
            )
        )

    if aggregate and ("public" in base_ensembles) and ("private" in base_ensembles):
        # merge per Qidx (public followed by private)
        merged = {}
        q_all = all_qindices([base_ensembles["public"], base_ensembles["private"]])
        for qi in q_all:
            merged[qi] = []
            merged[qi] += base_ensembles["public"].get(qi, [])
            merged[qi] += base_ensembles["private"].get(qi, [])
        # pick queries/refs: prefer public if available else private
        # (queries/refs are aligned by position, but we only use qi index access below)
        queries_acc = (
            base_queries["public"]
            if "public" in base_queries
            else base_queries.get("private", [])
        )
        refs_acc = (
            base_refs["public"]
            if "public" in base_refs
            else base_refs.get("private", [])
        )
        work_items.append(("public+private", merged, queries_acc, refs_acc))

    # --------------------------
    # 3) For each item, export embeddings (+ optional clustering)
    # --------------------------
    subname = os.path.basename(filename_base)
    batch_size = 64
    concurrency = 4

    async with aiohttp.ClientSession() as session:
        for split_tag, merged, queries_acc, refs_acc in work_items:
            # output dir per split
            if ood:
                npy_dir = f"semantic_group/ood_{subname}_{split_tag}"
            else:
                npy_dir = f"semantic_group/{subname}_{split_tag}"
            os.makedirs(npy_dir, exist_ok=True)
            print(f"[EMB] Export directory: {npy_dir}")

            # compute embeddings per Q
            semaphore = asyncio.Semaphore(concurrency)
            memory_embeddings: Dict[int, np.ndarray] = {}

            # stable Q order
            q_indices = sorted(merged.keys())
            for qi in tqdm(q_indices, desc=f"Embedding per Q [{split_tag}]"):
                pred_list = merged[qi]
                texts = [p[0] for p in pred_list]  # embed text only
                vecs: List[List[float]] = []
                for b in range(0, len(texts), batch_size):
                    batch = texts[b : b + batch_size]
                    if len(batch) == 0:
                        continue
                    vecs_batch = await _embed_batch(
                        session, batch, embedding_model, semaphore
                    )
                    vecs.extend(vecs_batch)
                arr = (
                    np.array(vecs, dtype=np.float32)
                    if vecs
                    else np.empty((0, 0), dtype=np.float32)
                )
                memory_embeddings[qi] = arr

                if export_embedding:
                    out_path = os.path.join(npy_dir, f"{qi}.npy")
                    with open(out_path, "wb") as f:
                        np.save(f, arr)

            print(
                f"[EMB] Completed embeddings for {len(q_indices)} questions ({split_tag})."
            )

            # Optional clustering per split
            if do_cluster:
                print(
                    f"[CLUSTER] Building semantic groups by cosine similarity threshold... ({split_tag})"
                )
                from sklearn.metrics.pairwise import cosine_similarity

                def cluster_by_threshold(embs: np.ndarray, thr: float):
                    if embs.size == 0 or embs.shape[0] <= 1:
                        # single or no candidate -> one cluster or empty
                        if embs.shape[0] == 0:
                            return [], {}
                        return [0], {0: [0]}
                    sim = cosine_similarity(embs)
                    n = sim.shape[0]
                    groups = [-1] * n
                    clusters: Dict[int, List[int]] = {}
                    gid = 0
                    for i in range(n):
                        if groups[i] != -1:
                            continue
                        groups[i] = gid
                        clusters[gid] = [i]
                        for j in range(i + 1, n):
                            if groups[j] != -1:
                                continue
                            if sim[i, j] >= thr:
                                groups[j] = gid
                                clusters[gid].append(j)
                        gid += 1
                    return groups, clusters

                export_json = []
                for qi in tqdm(
                    sorted(memory_embeddings.keys()),
                    desc=f"Clustering per Q [{split_tag}]",
                ):
                    embs = memory_embeddings[qi]
                    groups, clusters = cluster_by_threshold(embs, similarity_threshold)
                    # pick representative (closest to centroid)
                    reps = {}
                    for gid, idxs in clusters.items():
                        if len(idxs) == 1:
                            reps[gid] = idxs[0]
                        else:
                            centroid = embs[idxs].mean(axis=0, keepdims=True)
                            dists = np.linalg.norm(embs[idxs] - centroid, axis=1)
                            reps[gid] = idxs[int(np.argmin(dists))]

                    items = merged[qi]
                    grouped = []
                    for gid, idxs in clusters.items():
                        if not idxs:
                            continue
                        grouped.append(
                            {
                                "group_id": int(gid),
                                "members": [
                                    {
                                        "text": items[k][0],
                                        "tag": items[k][1],
                                        "index": int(k),
                                    }
                                    for k in idxs
                                ],
                                "representative": {
                                    "text": items[reps[gid]][0],
                                    "tag": items[reps[gid]][1],
                                    "index": int(reps[gid]),
                                },
                            }
                        )

                    export_json.append(
                        {
                            "Qidx": int(qi),
                            "query": (queries_acc[qi] if qi < len(queries_acc) else ""),
                            "reference": (refs_acc[qi] if qi < len(refs_acc) else None),
                            "num_candidates": int(len(items)),
                            "clusters": grouped,
                        }
                    )

                out_json = os.path.join(
                    npy_dir,
                    f"{subname}_{split_tag}_clusters_thr{similarity_threshold:.2f}.json",
                )
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(export_json, f, indent=2, ensure_ascii=False)
                print(f"[CLUSTER] Saved: {out_json}")


# ---------- CLI ----------
def build_argparser():
    p = argparse.ArgumentParser(
        description="End-to-end generation + semantic embedding/clustering pipeline"
    )

    # Generation args (from run.py, cleaned)
    p.add_argument("--data_name", type=str, default="knkarthick/samsum")
    p.add_argument("--subset", type=str, default=None)
    p.add_argument("--model_name", type=str, default="davinci-002")
    p.add_argument("--max_token", type=int, default=200)
    p.add_argument("--ice_num", type=int, default=0)
    p.add_argument("--ensemble", type=int, default=1)
    p.add_argument("--ds_size", type=int, default=100)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--ood", action="store_true", help="Use OOD test set")
    p.add_argument("--ood_subset", type=str, default=None)

    # Stage toggles
    p.add_argument(
        "--skip_gen",
        action="store_true",
        help="Skip generation stage and only run embeddings/clustering on existing JSONs",
    )
    p.add_argument(
        "--skip_embed", action="store_true", help="Skip embeddings/clustering stage"
    )

    # Embedding/clustering args (from semantic_cluster.py, cleaned)
    p.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-ada-002",
        help="OpenAI Embeddings model id (e.g., text-embedding-3-small / text-embedding-3-large / text-embedding-ada-002)",
    )
    p.add_argument(
        "--query_type",
        type=str,
        default="summary",
        choices=["summary", "QA"],
        help="How to parse the 'query' out of the prompt for export",
    )
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="Use public ensembles (else private if available)",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Fraction to subsample ensembles (0<gamma<=1)",
    )
    p.add_argument(
        "--export_embedding", action="store_true", help="Export per-Q .npy embeddings"
    )
    p.add_argument(
        "--cluster",
        action="store_true",
        help="Also compute simple cosine-threshold clustering",
    )
    p.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.90,
        help="Cosine similarity threshold for grouping",
    )
    p.add_argument(
        "--emb_splits",
        type=str,
        default="both",
        choices=["public", "private", "both"],
        help="Which split(s) to embed/export: public, private, or both (default). "
        "Overrides --aggregate for the embedding stage.",
    )

    return p


async def main_async(args):
    # Stage GEN
    if not args.skip_gen:
        filename_base, splits = await run_generation_stage(args)
    else:
        # When skipping, we need to infer a filename_base from available files in ./output
        # For simplicity, ask user to supply the base; but since we must not ask, try to guess newest base
        # We will scan output dir for *_public.json or *_private.json and pick the most recent stem.
        outdir = "output"
        cands = []
        if os.path.isdir(outdir):
            for fn in os.listdir(outdir):
                if fn.endswith("_public.json") or fn.endswith("_private.json"):
                    cands.append(os.path.join(outdir, fn))
        if not cands:
            raise FileNotFoundError(
                "skip_gen 지정 시, output/ 안에 기존 *_public.json 또는 *_private.json 이 있어야 합니다."
            )
        latest = max(cands, key=lambda p: os.path.getmtime(p))
        stem = (
            latest[: -len("_public.json")]
            if latest.endswith("_public.json")
            else latest[: -len("_private.json")]
        )
        filename_base = stem
        splits = ["public"] if os.path.exists(f"{filename_base}_public.json") else []
        if os.path.exists(f"{filename_base}_private.json"):
            splits.append("private")
        print(f"[SKIP GEN] Using existing base: {filename_base} with splits={splits}")

    # Stage EMB/CLUSTER
    if not args.skip_embed:
        await compute_and_export_embeddings_per_q(
            filename_base=filename_base,
            splits=splits,
            query_type=args.query_type,
            embedding_model=args.embedding_model,
            aggregate=args.aggregate,
            gamma=args.gamma,
            export_embedding=args.export_embedding,
            similarity_threshold=args.similarity_threshold,
            do_cluster=args.cluster,
            ood=args.ood,
        )
    else:
        print("[SKIP EMB] Embedding/Clustering stage skipped.")


if __name__ == "__main__":
    args = build_argparser().parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
