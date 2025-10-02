#!/usr/bin/env bash
# run_e2e.sh
# End-to-End: generation → embedding (+ optional clustering)
# chmod +x run_e2e.sh
# ./run_e2e.sh qa_e2e
# ./run_e2e.sh sum_e2e
# ./run_e2e.sh qa_e2e_ood

set -euo pipefail

# ====== COMMON ======
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"     # needed for model calls in generation stage
EMBED_MODEL="text-embedding-3-small"
SIM_THR="0.90"
DS_SIZE=20
ENSEMBLE=15
ICE_NUM=4
MAX_TOKEN=512
TEMP=0.02

# ====== DATASETS / MODELS ======
# QA
QA_DATA="Malikeh1375/medical-question-answering-datasets"
QA_SUBSET="chatdoctor_icliniq"
QA_MODEL="gpt-3.5-turbo"

# Summarization
SUM_DATA="knkarthick/samsum"   # samsum mirror
SUM_MODEL="davinci-002"

# ====== HELPERS ======
qa_e2e() {
  # 1) generation (public & private)
  python run.py \
    --data_name "${QA_DATA}" \
    --subset "${QA_SUBSET}" \
    --model_name "${QA_MODEL}" \
    --ice_num ${ICE_NUM} \
    --ensemble ${ENSEMBLE} \
    --ds_size ${DS_SIZE} \
    --max_token ${MAX_TOKEN} \
    --temp ${TEMP}

  # 2) embeddings (+clusters)
  python run.py \
    --skip_gen \
    --embedding_model "${EMBED_MODEL}" \
    --export_embedding \
    --cluster --similarity_threshold ${SIM_THR}
}

sum_e2e() {
  python run.py \
    --data_name "${SUM_DATA}" \
    --model_name "${SUM_MODEL}" \
    --ice_num ${ICE_NUM} \
    --ensemble ${ENSEMBLE} \
    --ds_size ${DS_SIZE} \
    --max_token ${MAX_TOKEN} \
    --temp ${TEMP}

  python run.py \
    --skip_gen \
    --embedding_model "${EMBED_MODEL}" \
    --export_embedding \
    --cluster --similarity_threshold ${SIM_THR}
}

qa_e2e_ood() {
  # In-dist public, OOD test (subset name example: chatdoctor_healthcaremagic)
  OOD_SUBSET="chatdoctor_healthcaremagic"

  python run.py \
    --data_name "${QA_DATA}" \
    --model_name "${QA_MODEL}" \
    --ice_num ${ICE_NUM} \
    --ensemble ${ENSEMBLE} \
    --ds_size ${DS_SIZE} \
    --max_token ${MAX_TOKEN} \
    --temp ${TEMP} \
    --ood --ood_subset "${OOD_SUBSET}"

  python run.py \
    --skip_gen \
    --embedding_model "${EMBED_MODEL}" \
    --export_embedding \
    --cluster --similarity_threshold ${SIM_THR} \
    --ood
}

sum_e2e_ood() {
  OOD_SUBSET="knkarthick/dialogsum""
  
  python run.py \
    --data_name "${SUM_DATA}" \
    --model_name "${SUM_MODEL}" \
    --ice_num ${ICE_NUM} \
    --ensemble ${ENSEMBLE} \
    --ds_size ${DS_SIZE} \
    --max_token ${MAX_TOKEN} \
    --temp ${TEMP}
    -ood --ood_subset "${OOD_SUBSET}"

  python run.py \
    --skip_gen \
    --embedding_model "${EMBED_MODEL}" \
    --export_embedding \
    --cluster --similarity_threshold ${SIM_THR}
    -ood
}

# ====== DISPATCH ======
cmd="${1:-help}"
case "$cmd" in
  qa_e2e)       qa_e2e ;;
  sum_e2e)      sum_e2e ;;
  qa_e2e_ood)   qa_e2e_ood ;;
  sum_e2e_ood)  sum_e2e_ood ;;
  help|*) cat <<'EOF'
Usage:
  ./run_e2e.sh <command>

Commands:
  qa_e2e       # QA dataset: generation → embedding+clusters
  sum_e2e      # Summarization dataset: generation → embedding+clusters
  qa_e2e_ood   # QA OOD setting: generation → embedding+clusters (with --ood)
  sum_e2e_ood # Summarization OOD setting: eneration → embedding+clusters (with --ood) (*TBU)
EOF
  ;;
esac
