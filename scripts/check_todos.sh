#!/usr/bin/env bash
# Fail if any student TODO marker is left in the implementation.
set -euo pipefail

if grep -R "TODO(student)" -n src tests; then
  echo "ERROR: unfinished TODO(student) markers above." >&2
  exit 1
fi

echo "OK: no TODO(student) left in src/ or tests/."
