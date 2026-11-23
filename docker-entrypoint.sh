#!/bin/sh
# Switch between training and serving: first argument (or MODE env var) = train | serve
set -e

MODE="${1:-${MODE:-serve}}"
[ $# -gt 0 ] && shift

case "$MODE" in
  serve)
    exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-8080}"
    ;;
  train)
    exec python src/train.py "$@"
    ;;
  dummy-model)
    exec python scripts/make_dummy_model.py "$@"
    ;;
  *)
    echo "Unknown mode '$MODE'. Use: serve | train | dummy-model" >&2
    exit 1
    ;;
esac
