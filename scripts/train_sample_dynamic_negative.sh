#!/usr/bin/env bash
set -euo pipefail
exec python "$(dirname "$0")/train.py" --method sample-dynamic-negative "$@"
