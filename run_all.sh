#!/usr/bin/env bash
# ==============================================================================
# DVD Project (16nm LP CMOS Parity8) - Root Pipeline Wrapper
# Forwards all arguments to the canonical runner in regenerate_code/run_all.sh.
# ==============================================================================
set -euo pipefail

# 1. Resolve project root reliably using wrapper's own location
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# 2. Resolve canonical runner path
CANONICAL_RUNNER="${PROJECT_ROOT}/regenerate_code/run_all.sh"

if [ ! -f "${CANONICAL_RUNNER}" ]; then
    echo "Error: Canonical runner not found at: ${CANONICAL_RUNNER}" >&2
    exit 1
fi

if [ ! -x "${CANONICAL_RUNNER}" ]; then
    echo "Error: Canonical runner is not executable: ${CANONICAL_RUNNER}" >&2
    exit 1
fi

# 3, 4, 5. Forward all arguments exactly and preserve exit status using exec
exec "${CANONICAL_RUNNER}" "$@"
