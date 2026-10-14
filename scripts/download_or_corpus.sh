#!/usr/bin/env bash
# Port of OpenResearcher/scripts/download_corpus.sh: pull the OpenResearcher corpus parquet
# shards (~30 GB) into the Hugging Face cache ($HF_HOME). start_or_search_service.sh and
# build_or_bm25_index.sh resolve the snapshot from the cache with local_files_only=True.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Optional: OR_SERVICE_ENV = prefix of the search-service Python env (put first on PATH).
if [[ -n "${OR_SERVICE_ENV:-}" ]]; then export PATH="${OR_SERVICE_ENV}/bin:${PATH}"; fi

: "${HF_HOME:?Set and export HF_HOME to your Hugging Face cache root}"
command -v hf >/dev/null 2>&1 || {
  echo "error: hf is not available; activate the service env or set OR_SERVICE_ENV (pip install huggingface_hub)" >&2
  exit 1
}

corpus_dir="$(
  hf download OpenResearcher/OpenResearcher-Corpus --repo-type dataset \
    --include "data/*.parquet"
)"

echo "OpenResearcher corpus snapshot: ${corpus_dir}"
