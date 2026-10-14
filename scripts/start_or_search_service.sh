#!/usr/bin/env bash
# Start the OpenResearcher BM25 search service (src/server/openresearcher_env/deploy_search_service.py)
# over the OpenResearcher corpus.
#
#   bash scripts/start_or_search_service.sh PORT [URL_FILE]
#     PORT      uvicorn port
#     URL_FILE  where to write "http://<host>:<port>" once the service answers GET / (optional)
#
# The service loads the whole corpus into RAM before it answers (~10+ minutes, ~128 GB of memory);
# this script backgrounds uvicorn, polls GET / until the corpus is loaded, THEN publishes the URL,
# and waits on the server so it lives as long as the server does.
#
# Required env: HF_HOME (HF cache holding the corpus snapshot: scripts/download_or_corpus.sh),
# JAVA_HOME (JDK 21), OR_WORK_ROOT (holds lucene-jars/ and bm25/; scratch goes to $OR_WORK_ROOT/tmp).
# Optional: OR_INDEX_DIR [$OR_WORK_ROOT/bm25], LUCENE_EXTRA_DIR [$OR_WORK_ROOT/lucene-jars],
# OR_START_TIMEOUT_S [1200], MAX_SNIPPET_LEN [180, read by the app], OR_SERVICE_ENV [prefix of the
# service Python env, put first on PATH].
#
# Read-only on every asset: the corpus snapshot, the index and the jars are only checked for
# existence and handed to the JVM; nothing is created outside $OR_WORK_ROOT/tmp and URL_FILE.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SELF_DIR/../src/server/openresearcher_env" && pwd)"
if [[ -n "${OR_SERVICE_ENV:-}" ]]; then export PATH="${OR_SERVICE_ENV}/bin:${PATH}"; fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

PORT="${1:?usage: start_or_search_service.sh PORT [URL_FILE]}"
URL_FILE="${2-}"

: "${HF_HOME:?Export HF_HOME (Hugging Face cache holding the corpus snapshot)}"
: "${JAVA_HOME:?Export JAVA_HOME for JDK 21}"
: "${OR_WORK_ROOT:?Export OR_WORK_ROOT (root holding lucene-jars/ and bm25/)}"
OR_WORK_ROOT="${OR_WORK_ROOT%/}"
LUCENE_EXTRA_DIR="${LUCENE_EXTRA_DIR:-${OR_WORK_ROOT}/lucene-jars}"
index_dir="${OR_INDEX_DIR:-${OR_WORK_ROOT}/bm25}"
temp_dir="${OR_WORK_ROOT}/tmp"

mkdir -p "${temp_dir}"
export TMPDIR="${temp_dir}"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:+${JAVA_TOOL_OPTIONS} }-Djava.io.tmpdir=${temp_dir}"
export JVM_PATH="${JAVA_HOME}/lib/server/libjvm.so"
export PATH="${JAVA_HOME}/bin:${PATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-unused}"   # parity with the upstream launcher; the app never reads it

# Resolve the corpus snapshot from the HF cache (no network: local_files_only).
corpus_dir="$(
python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    "OpenResearcher/OpenResearcher-Corpus",
    repo_type="dataset",
    allow_patterns="data/*.parquet",
    local_files_only=True,
))
PY
)"

if ! compgen -G "${corpus_dir}/data/*.parquet" >/dev/null; then
  echo "error: no corpus parquet files found in ${corpus_dir}/data; run scripts/download_or_corpus.sh first" >&2
  exit 1
fi
for jar in lucene-highlighter lucene-queries lucene-memory; do
  if [[ ! -f "${LUCENE_EXTRA_DIR}/${jar}-9.9.1.jar" ]]; then
    echo "error: Lucene JAR not found: ${LUCENE_EXTRA_DIR}/${jar}-9.9.1.jar; run scripts/download_or_jars.sh first" >&2
    exit 1
  fi
done
if [[ ! -d "${index_dir}" ]]; then
  echo "error: BM25 index not found at ${index_dir}; run scripts/build_or_bm25_index.sh first" >&2
  exit 1
fi
if [[ ! -f "${JVM_PATH}" ]]; then
  echo "error: libjvm.so not found at ${JVM_PATH} (JAVA_HOME=${JAVA_HOME} must be a JDK 21)" >&2
  exit 1
fi

cd "${OR_WORK_ROOT}"
export SEARCHER_TYPE=bm25
export CORPUS_PARQUET_PATH="${corpus_dir}/data/*.parquet"
export LUCENE_INDEX_DIR="${index_dir}"
export LUCENE_EXTRA_DIR

echo "[or-service] parquet=${CORPUS_PARQUET_PATH}"
echo "[or-service] index=${LUCENE_INDEX_DIR} jars=${LUCENE_EXTRA_DIR}"

python -m uvicorn --app-dir "${APP_DIR}" deploy_search_service:app --host 0.0.0.0 --port "${PORT}" &
server_pid=$!

cleanup() {
  status=$?
  trap - EXIT
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

timeout_s="${OR_START_TIMEOUT_S:-1200}"
interval_s=5
elapsed=0
ready=0
while (( elapsed < timeout_s )); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "error: search service exited during startup" >&2
    wait "${server_pid}" || true
    exit 1
  fi
  if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep "${interval_s}"
  elapsed=$(( elapsed + interval_s ))
done
if (( ready == 0 )); then
  echo "error: search service did not answer GET / within ${timeout_s}s" >&2
  exit 1
fi

service_url="http://$(hostname -s):${PORT}"
if [[ -n "${URL_FILE}" ]]; then
  mkdir -p "$(dirname "${URL_FILE}")"
  tmp_file="${URL_FILE}.tmp.$$"
  printf '%s\n' "${service_url}" > "${tmp_file}"
  mv -f "${tmp_file}" "${URL_FILE}"
  echo "[or-service] wrote ${URL_FILE}"
fi
echo "OR_SERVICE_URL=${service_url} ready after ${elapsed}s"

wait "${server_pid}"
