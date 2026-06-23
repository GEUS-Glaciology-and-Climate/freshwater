#!/usr/bin/env bash
# Partition MAR runoff across drainage basins.
#
# Outputs ./dat/MAR_runoff_{ice|land}_{year}_100.bsv for each year (1940-2026).
# Skips years whose BSV file already exists.
#
# Also builds the shared BSV column headers (./tmp/header_{ice|land}_{k}.bsv)
# used by all RCM partitioning scripts. Run this before rcm_racmo.sh.

set -o nounset
set -o pipefail

# set -o errexit
### uncomment above to help find errors. When rerunning and counting on
### GRASS failing w/ overwrite issues for speed, leave it commented.

red='\033[0;31m'; orange='\033[0;33m'; green='\033[0;32m'; nc='\033[0m'
log_info() { echo -e "${green}[$(date --iso-8601=seconds)] [INFO] ${@}${nc}"; }
log_warn() { echo -e "${orange}[$(date --iso-8601=seconds)] [WARN] ${@}${nc}"; }
log_err()  { echo -e "${red}[$(date --iso-8601=seconds)] [ERR] ${@}${nc}" >&2; }

trap ctrl_c INT
function ctrl_c() {
  log_warn "Caught CTRL-C, killing process"
  kill -term $$
}

export GRASS_VERBOSE=3
unset LD_LIBRARY_PATH
export PROJ_LIB=/usr/share/proj

if [ -z ${DATADIR+x} ]; then
    echo "DATADIR environment variable is unset."
    echo "Fix with: \"export DATADIR=/path/to/data\""
    exit 255
fi

set -x

# Setup MAR mapset
# For the ice domain routed to the coast, 1 km^2 model grid cells that only
# partially cover an ice basin are included in their entirety. Build a basin
# mask and apply AND(basin_mask, model_mask) to limit to the smallest ice subset.
log_info "Initializing MAR mapset..."

g.mapset MAR

g.region -d
r.mapcalc "mask_ice_basin      = if(basins@ice_100, 1, null())"
r.mapcalc "mask_land_basin     = if(not(isnull(basins@land_100)) & isnull(mask_ice_basin), 1, null())"
r.mapcalc "mask_ice_land_basin = if(not(isnull(basins_filled@land_100)), 1, null())"

g.region MAR@MAR -p
r.in.gdal -o input="NetCDF:${DATADIR}/MAR/3.14-daily-1km/MARv3.14-daily-ERA5-2000.nc:MSK" output=MSK
r.region map=MSK region=MAR@MAR
r.mapcalc "mask_ice_MAR  = if((MSK >= 2) & mask_ice_land_basin, 1, null())"
r.mapcalc "mask_land_MAR = if((MSK == 1) & mask_ice_land_basin, 1, null())"
g.region -d

# basin_partition: extract one time step from a NetCDF and compute per-basin sum
function basin_partition() {
    JOB="$1"      # 0 to ndays-1
    SLOT="$2"     # 0 to ncpus-1
    ZONES="$3"    # e.g. basins@land_100
    RCMfilevar=$4 # e.g. MAR_runoff_ice_2000.nc:runoff
    REGION=$5     # e.g. MAR@MAR

    RCMfile=${RCMfilevar/:*/}

    r.external -o source="netCDF:tmp/${RCMfilevar}" output=model_${SLOT} band=${JOB} --o --q 2>/dev/null
    r.region map=model_${SLOT} region=${REGION} --q
    DATESTR0=$(ncdump -h tmp/${RCMfile} | grep time:units)
    DATESTR0=${DATESTR0: 27:10}
    DATESTR=$(date -d"${DATESTR0}+$(( ${JOB}-1 )) days" +"%Y-%m-%d")
    r.univar -t map=model_${SLOT} zones=${ZONES} --q 2>/dev/null | cut -d"|" -f13 | datamash -t"|" transpose | sed s/^sum/${DATESTR}/
}
export -f basin_partition

# Build shared BSV column headers (basin cat IDs) — used by all RCM scripts
log_info "Building BSV headers..."

r.mapcalc "model = 1" --o  # dummy raster to drive r.univar
for k_pct in 100 90 80; do
    r.univar -t map=model zones=basins@ice_${k_pct} \
        | sed s/^zone/cat/ \
        | cut -d"|" -f1 \
        | datamash -t"|" transpose \
        > ./tmp/header_ice_${k_pct}.bsv
done

k_pct=100
r.univar -t map=model zones=basins@land_${k_pct} \
    | sed s/^zone/cat/ \
    | cut -d"|" -f1 \
    | datamash -t"|" transpose \
    > ./tmp/header_land_${k_pct}.bsv

# MAR ice runoff (1940-2026)
log_info "Using EXTRA GNU parallel options via \${PARALLEL}: ${PARALLEL:-Not set}"

k_pct=100
for y in $(seq 1940 2026); do
    file_base=MAR_runoff_ice_${y}
    infile=${file_base}.nc
    outfile=${file_base}_${k_pct}.bsv

    log_info "MAR ice: ${outfile}"

    if [[ -e ./dat/${outfile} ]]; then
        log_warn "${outfile} exists. Skipping..."
    else
        cp ./tmp/header_ice_${k_pct}.bsv ./dat/${outfile}
        NT=$(ncdump -h ./tmp/${infile} | sed -n 's/.*time = \([0-9]\+\) ;/\1/p')
        seq 0 $((NT - 1)) \
            | parallel --nice 9 --keep-order --bar \
                basin_partition {#} {%} basins@ice_${k_pct} "${infile}:runoff" MAR@MAR \
            >> ./dat/${outfile}
    fi
done

# MAR land runoff (1940-2026)
k_pct=100
for y in $(seq 1940 2026); do
    file_base=MAR_runoff_land_${y}
    infile=${file_base}.nc
    outfile=${file_base}_${k_pct}.bsv

    log_info "MAR land: ${outfile}"

    if [[ -e ./dat/${outfile} ]]; then
        log_warn "${outfile} exists. Skipping..."
    else
        cp ./tmp/header_land_${k_pct}.bsv ./dat/${outfile}
        NT=$(ncdump -h ./tmp/${infile} | sed -n 's/.*time = \([0-9]\+\) ;/\1/p')
        seq 0 $((NT - 1)) \
            | parallel --nice 9 --keep-order --bar \
                basin_partition {#} {%} basins@land_${k_pct} "${infile}:runoff" MAR@MAR \
            >> ./dat/${outfile}
    fi
done
