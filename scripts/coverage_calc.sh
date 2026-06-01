#!/usr/bin/env bash
# Calculate RCM coverage fraction for each drainage basin.
#
# Coverage = fraction of each basin's area that is covered by model grid cells.
# Applied later in bsv2netcdf.py to scale discharge by coverage.

set -o nounset
set -o pipefail

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
export PROJ_LIB=/usr/share/proj

if [ -z ${DATADIR+x} ]; then
    echo "DATADIR environment variable is unset."
    echo "Fix with: \"export DATADIR=/path/to/data\""
    exit 255
fi

set -x

# Ice basins: coverage by MAR and RACMO ice masks
for k in 80 90 100; do
  mapset=ice_${k}
  g.mapset ${mapset}

  r.mask -r
  r.stats --q -aN input=basins@${mapset},mask_ice_MAR@MAR separator=,  \
	  > ./tmp/coverage_MAR_ice_${k}.csv
  r.stats --q -aN input=basins@${mapset},mask_ice_RACMO@RACMO separator=,  \
	  > ./tmp/coverage_RACMO_ice_${k}.csv
done

# Land basins: coverage by MAR and RACMO land masks
k=100
mapset=land_${k}
g.mapset ${mapset}

r.mask -r
r.stats --q -aN input=basins@${mapset},mask_land_MAR@MAR separator=,  \
	> ./tmp/coverage_MAR_land_${k}.csv
r.stats --q -aN input=basins@${mapset},mask_land_RACMO@RACMO separator=,  \
	> ./tmp/coverage_RACMO_land_${k}.csv
