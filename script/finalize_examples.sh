#!/usr/bin/env bash
# finalize_examples.sh
# Finalize: DPM/KSA -> LLM ALWAYS chooses the final answer/summary.

set -euo pipefail

# ====== COMMON (override via env) ======
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"        # required for LLM calls
TOPK="${TOPK:-40}"                                  # candidates passed to LLM
MAX_TOKEN="${MAX_TOKEN:-512}"
TEMP="${TEMP:-0.02}"
SUBSET="${SUBSET:-chatdoctor_icliniq}"             # QA subset (optional in STEM)
SUM_SUBSET="${SUM_SUBSET:-}"                        # Summarization subset (optional in STEM)
ENSEMBLE="${ENSEMBLE:-15}"
ICE_NUM="${ICE_NUM:-4}"

# ====== DP / AGGREGATION KNOBS ======
DPM_EPS="${DPM_EPS:-1.0}"
DPM_EPS_EM="${DPM_EPS_EM:-0.12}"
DPM_EPS_GM="${DPM_EPS_GM:-7.51}"
DPM_LEVELS="${DPM_LEVELS:-4}"
DPM_POOL="${DPM_POOL:-auto}"                       # auto|public|private|all

KSA_EPS_QA="${KSA_EPS_QA:-1.25}"
KSA_K_QA="${KSA_K_QA:-60}"

KSA_EPS_SUM="${KSA_EPS_SUM:-1.25}"
KSA_K_SUM="${KSA_K_SUM:-40}"

# ====== STEMS (must match E2E outputs) ======
QA_MODEL="${QA_MODEL:-gpt-3.5-turbo}"
QA_STEM="medical-question-answering-datasets${SUBSET:+_${SUBSET}}_${QA_MODEL}_${ENSEMBLE}way-${ICE_NUM}shot"
QA_BASE="output/${QA_STEM}"

SUM_MODEL="${SUM_MODEL:-davinci-002}"
SUM_STEM="samsum${SUM_SUBSET:+_${SUM_SUBSET}}_${SUM_MODEL}_${ENSEMBLE}way-${ICE_NUM}shot"
SUM_BASE="output/${SUM_STEM}"

# ====== QA FINALIZATION ======
qa_dpm() {
  python finalize_output.py \
    --filename_base "${QA_BASE}" \
    --split public+private \
    --task qa \
    --method dpm \
    --topk "${TOPK}" \
    --dpm_eps "${DPM_EPS}" --dpm_eps_em "${DPM_EPS_EM}" --dpm_eps_gm "${DPM_EPS_GM}" --dpm_levels "${DPM_LEVELS}" \
    --dpm_pool "${DPM_POOL}" \
    --model "${QA_MODEL}" --max_tokens "${MAX_TOKEN}" --temperature "${TEMP}"
}

qa_ksa() {
  python finalize_output.py \
    --filename_base "${QA_BASE}" \
    --split public \
    --task qa \
    --method ksa \
    --topk "${TOPK}" \
    --ksa_eps "${KSA_EPS_QA}" --ksa_k "${KSA_K_QA}" --ksa_source public+private \
    --model "${QA_MODEL}" --max_tokens "${MAX_TOKEN}" --temperature "${TEMP}"
}

qa_dpm_ood() {
  python finalize_output.py \
    --filename_base "${QA_BASE}" \
    --split public \
    --task qa \
    --method dpm \
    --topk "${TOPK}" \
    --dpm_eps "${DPM_EPS}" --dpm_eps_em "${DPM_EPS_EM}" --dpm_eps_gm "${DPM_EPS_GM}" --dpm_levels "${DPM_LEVELS}" \
    --dpm_pool "${DPM_POOL}" \
    --model "${QA_MODEL}" --max_tokens "${MAX_TOKEN}" --temperature "${TEMP}" \
    --ood
}

# ====== SUMMARIZATION FINALIZATION ======
sum_dpm() {
  python finalize_output.py \
    --filename_base "${SUM_BASE}" \
    --split public+private \
    --task summarization \
    --method dpm \
    --topk "${TOPK}" \
    --dpm_eps "${DPM_EPS}" --dpm_eps_em "${DPM_EPS_EM}" --dpm_eps_gm "${DPM_EPS_GM}" --dpm_levels "${DPM_LEVELS}" \
    --dpm_pool "${DPM_POOL}" \
    --model "${QA_MODEL}" --max_tokens "${MAX_TOKEN}" --temperature "${TEMP}"
}

sum_ksa() {
  python finalize_output.py \
    --filename_base "${SUM_BASE}" \
    --split public \
    --task summarization \
    --method ksa \
    --topk "${TOPK}" \
    --ksa_eps "${KSA_EPS_SUM}" --ksa_k "${KSA_K_SUM}" --ksa_source public+private \
    --model "${QA_MODEL}" --max_tokens "${MAX_TOKEN}" --temperature "${TEMP}"
}

# ====== DISPATCH ======
cmd="${1:-help}"
case "$cmd" in
  qa_dpm)        qa_dpm ;;
  qa_ksa)        qa_ksa ;;
  qa_dpm_ood)    qa_dpm_ood ;;
  sum_dpm)       sum_dpm ;;
  sum_ksa)       sum_ksa ;;
  help|*) cat <<'EOF'
Usage:
  ./finalize_examples.sh <command>

Commands:
  qa_dpm       # QA: build candidates via DPM, LLM selects 
  qa_ksa       # QA: build hints via KSA, LLM selects 
  qa_dpm_ood   # QA: DPM → LLM on OOD split
  sum_dpm      # Summarization: DPM → LLM 
  sum_ksa      # Summarization: KSA → LLM   

Notes:
  - LLM always makes the final decision/generation.
  - STEMs must match your E2E outputs; override SUBSET/SUM_SUBSET/ENSEMBLE/ICE_NUM if needed.
  - You can override any var inline, e.g.:
      DPM_EPS=2.0 TOPK=5 ./finalize_examples.sh qa_dpm
EOF
  ;;
esac