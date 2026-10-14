#!/usr/bin/env bash
# Port of OpenResearcher/scripts/download_jars.sh: fetch the Lucene 9.9.1 highlighter/queries/memory
# jars that deploy_search_service.py adds to the Anserini classpath (UnifiedHighlighter snippets).
#
# NEVER CLOBBERS. $LUCENE_EXTRA_DIR may point at jars provisioned by another project, so this
# script treats an existing jar as read-only:
# a jar that is already there is left untouched, and a missing one is fetched to a temp file in the
# same directory and moved into place only after the download succeeds. `wget -O <target>` -- what
# this script used to do -- truncates the target the instant it starts, so a node with no outbound
# network left 0-byte jars behind.
#
#   OR_FORCE=1 bash scripts/download_or_jars.sh    # re-download and replace jars that are already present
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Optional: OR_SERVICE_ENV = prefix of the search-service Python env (put first on PATH).
if [[ -n "${OR_SERVICE_ENV:-}" ]]; then export PATH="${OR_SERVICE_ENV}/bin:${PATH}"; fi

: "${OR_WORK_ROOT:?Set and export OR_WORK_ROOT (root for the Lucene jars, the BM25 index and scratch; large storage)}"
jar_dir="${LUCENE_EXTRA_DIR:-${OR_WORK_ROOT}/lucene-jars}"
force="${OR_FORCE:-0}"

if [[ "${jar_dir}" != "${OR_WORK_ROOT%/}/"* ]]; then
  echo "note: ${jar_dir} is outside OR_WORK_ROOT (${OR_WORK_ROOT}) -- treating its jars as someone else's" >&2
fi

mkdir -p "${jar_dir}"

fetch() {  # fetch URL DEST -- writes DEST only on success
  local url="$1" dest="$2" tmp
  tmp="$(mktemp "${dest}.part.XXXXXX")"
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "${tmp}" "${url}" || { rm -f -- "${tmp}"; return 1; }
  else
    curl -fL -o "${tmp}" "${url}" || { rm -f -- "${tmp}"; return 1; }
  fi
  [[ -s "${tmp}" ]] || { rm -f -- "${tmp}"; return 1; }
  mv -f -- "${tmp}" "${dest}"
}

kept=0 got=0
for jar in lucene-highlighter lucene-queries lucene-memory; do
  dest="${jar_dir}/${jar}-9.9.1.jar"
  if [[ -s "${dest}" && "${force}" != "1" ]]; then
    echo "keep     ${dest} ($(stat -c %s "${dest}") bytes)"
    kept=$(( kept + 1 ))
    continue
  fi
  url="https://repo1.maven.org/maven2/org/apache/lucene/${jar}/9.9.1/${jar}-9.9.1.jar"
  echo "download ${url}"
  if ! fetch "${url}" "${dest}"; then
    echo "error: failed to download ${url}; ${dest} left as it was" >&2
    exit 1
  fi
  got=$(( got + 1 ))
done

echo "OpenResearcher Lucene JARs: ${jar_dir} (${kept} kept, ${got} downloaded)"
