#!/usr/bin/env bash
# Calculate RCM coverage fraction for each drainage basin.
#
# Coverage = fraction of each basin's area that is covered by model grid cells.
# Applied later in bsv2netcdf.py to scale discharge by coverage.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

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
