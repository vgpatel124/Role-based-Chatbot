#!/bin/sh
# Index the documents into ChromaDB and BM25 unless that was already done.
#
# `python -m app.ingest` appends to existing collections, so running it twice
# duplicates every chunk. This wrapper makes it safe to run on every start:
#   - already indexed        -> skip
#   - missing or partial     -> wipe and index from scratch
#   - FORCE_INGEST=1         -> wipe and re-index (after changing documents)
set -eu

DB="${CHROMA_DB_PATH:?CHROMA_DB_PATH must be set}"

# ingest writes bm25_docs last, so its presence means the run completed.
if [ "${FORCE_INGEST:-0}" != "1" ] && ls "$DB"/bm25_docs/*.json >/dev/null 2>&1; then
    echo "Index already present in $DB - skipping (set FORCE_INGEST=1 to rebuild)."
    exit 0
fi

echo "Building index in $DB ..."
mkdir -p "$DB"
find "$DB" -mindepth 1 -delete
exec python -m app.ingest
