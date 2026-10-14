#!/usr/bin/env bash
# Port of OpenResearcher/scripts/build_bm25_index.sh: OpenResearcher corpus parquet
#   --duckdb--> corpus.jsonl {"id","contents"} --pyserini--> Lucene BM25 index at $OR_WORK_ROOT/bm25
# Needs: download_or_corpus.sh done, JDK 21, the service env active. Scratch lives under
# $OR_WORK_ROOT/tmp/build-bm25 and is removed on exit (success or failure). Tens of GB of scratch,
# a few hours on 8 threads (OR_INDEX_THREADS).
#
# The index goes to $OR_INDEX_DIR (default $OR_WORK_ROOT/bm25). OR_PROVISION=1 is required to
# actually build it. Refuses to run if $OR_INDEX_DIR is already populated -- an index someone else built is
# read-only to this repo. OR_FORCE=1 overrides.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Optional: OR_SERVICE_ENV = prefix of the search-service Python env (put first on PATH).
if [[ -n "${OR_SERVICE_ENV:-}" ]]; then export PATH="${OR_SERVICE_ENV}/bin:${PATH}"; fi

: "${HF_HOME:?Set and export HF_HOME to your Hugging Face cache root}"
: "${JAVA_HOME:?Set and export JAVA_HOME to a JDK 21 installation}"
: "${OR_WORK_ROOT:?Set and export OR_WORK_ROOT (root for the Lucene jars, the BM25 index and scratch; large storage)}"
command -v python >/dev/null 2>&1 || {
  echo "error: python is not available; activate the service env" >&2
  exit 1
}

temp_root="${OR_WORK_ROOT%/}/tmp"
work_root="${temp_root}/build-bm25"
index_dir="${OR_INDEX_DIR:-${OR_WORK_ROOT%/}/bm25}"
temp_dir="${work_root}/runtime"

cleanup() {
  status=$?
  trap - EXIT

  if [[ -n "${work_root}" && "${work_root}" == "${temp_root}/"* ]]; then
    if ! rm -rf -- "${work_root}"; then
      echo "error: failed to remove build scratch directory: ${work_root}" >&2
      status=1
    fi
  else
    echo "error: refusing to clean unexpected path: ${work_root}" >&2
    status=1
  fi

  exit "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -d "${index_dir}" ]] && compgen -G "${index_dir}/*" >/dev/null; then
  if [[ "${OR_FORCE:-0}" != "1" ]]; then
    echo "error: ${index_dir} already holds an index -- refusing to write into it." >&2
    echo "       Point OR_INDEX_DIR at an empty directory, or re-run with OR_FORCE=1 to add to it." >&2
    exit 1
  fi
  echo "warning: OR_FORCE=1 -- pyserini will write into the existing index at ${index_dir}" >&2
fi

# Creating a new asset tree is an explicit, opt-in act: say where it will land and require
# OR_PROVISION=1. Without this, a stray OR_WORK_ROOT silently starts a multi-GB download into
# whatever directory it happens to point at.
if [[ "${OR_PROVISION:-0}" != "1" ]]; then
  echo "This would CREATE an OpenResearcher BM25 index (tens of GB of scratch) under:"
  echo "    ${index_dir}"
  echo "Nothing has been written. Re-run with OR_PROVISION=1 if that path is correct."
  exit 1
fi

mkdir -p "${work_root}/collection" "${index_dir}" "${temp_dir}"
export TMPDIR="${temp_dir}"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:+${JAVA_TOOL_OPTIONS} }-Djava.io.tmpdir=${temp_dir}"
export PATH="${JAVA_HOME}/bin:${PATH}"
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
  echo "error: no corpus parquet files found in ${corpus_dir}/data" >&2
  echo "run scripts/download_or_corpus.sh first" >&2
  exit 1
fi

export OR_CORPUS_DIR="${corpus_dir}"
export OR_WORK_DIR="${work_root}"

echo "duckdb"
python - <<'PY'
import json
import os
import pathlib

import duckdb
from tqdm import tqdm

corpus = pathlib.Path(os.environ["OR_CORPUS_DIR"])
work = pathlib.Path(os.environ["OR_WORK_DIR"])
collection = work / "collection"
collection.mkdir(parents=True, exist_ok=True)
duckdb_temp = work / "runtime" / "duckdb"
duckdb_temp.mkdir(parents=True, exist_ok=True)

parquet_glob = str(corpus / "data" / "*.parquet")
connection = duckdb.connect(config={"temp_directory": str(duckdb_temp)})
total = connection.execute(
    "SELECT count(*) FROM read_parquet(?)",
    [parquet_glob],
).fetchone()[0]
rows = connection.execute(
    "SELECT docid, text FROM read_parquet(?)",
    [parquet_glob],
)
with (collection / "corpus.jsonl").open("w", encoding="utf-8") as output:
    with tqdm(
        total=total,
        desc="Converting corpus to JSONL",
        unit="doc",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        while batch := rows.fetchmany(10_000):
            for docid, text in batch:
                record = {"id": str(docid), "contents": text or ""}
                output.write(json.dumps(record) + "\n")
            progress.update(len(batch))
PY

cd "${work_root}"
echo "pyserini"
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "${work_root}/collection" \
  --index "${index_dir}" \
  --generator DefaultLuceneDocumentGenerator \
  --threads "${OR_INDEX_THREADS:-8}"

echo "OpenResearcher BM25 index: ${index_dir}"
