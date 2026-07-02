# Shared setup for all freshwater GRASS/bash stage scripts.
# Source this at the top of each script:
#     source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# It sets shell options, logging helpers, the GRASS/PROJ environment, and
# validates that DATADIR is set. It intentionally does NOT enable `set -x`
# or `set -o errexit` — each script decides that for itself.

set -o nounset
set -o pipefail

# NOTE: `set -o errexit` is deliberately left off. Several stages rerun GRASS
# commands that fail on purpose (e.g. overwrite conflicts used as skip logic).
# Prefer explicit `if [[ -e ... ]]; then skip` guards over global errexit.

red='\033[0;31m'; orange='\033[0;33m'; green='\033[0;32m'; nc='\033[0m'
log_info() { echo -e "${green}[$(date --iso-8601=seconds)] [INFO] ${@}${nc}"; }
log_warn() { echo -e "${orange}[$(date --iso-8601=seconds)] [WARN] ${@}${nc}"; }
log_err()  { echo -e "${red}[$(date --iso-8601=seconds)] [ERR] ${@}${nc}" >&2; }

trap ctrl_c INT
function ctrl_c() {
  log_warn "Caught CTRL-C, killing process"
  kill -term $$
}

debug() { if [[ debug:- == 1 ]]; then log_warn "debug:"; echo $@; fi; }

export GRASS_VERBOSE=3
unset LD_LIBRARY_PATH
# Override PROJ_LIB in the environment if your install differs.
export PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
# Python for stages that shell out to it; Makefile exports the env's interpreter.
PYTHON="${PYTHON:-python}"

if [ -z ${DATADIR+x} ]; then
    log_err "DATADIR environment variable is unset."
    log_err "Fix with: \"export DATADIR=/path/to/data\""
    exit 255
fi
