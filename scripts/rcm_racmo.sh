#!/usr/bin/env bash
# Partition RACMO runoff across drainage basins.
#
# Outputs ./dat/RACMO_runoff_{ice|land}_{year}_100.bsv for each year (1958-2024).
# Skips years whose BSV file already exists.
#
# Depends on BSV column headers written by rcm_mar.sh. Run rcm_mar.sh first.

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

# BSV headers are written by rcm_mar.sh — fail early if missing
if [[ ! -f ./tmp/header_ice_100.bsv ]]; then
    log_err "BSV headers not found in ./tmp/. Run rcm_mar.sh first."
    exit 1
fi

set -x

# Setup RACMO mapset
log_info "Initializing RACMO mapset..."

g.mapset -c RACMO

g.region -d
r.mapcalc "mask_ice_basin      = if(basins@ice_100, 1, null())"
r.mapcalc "mask_land_basin     = if(not(isnull(basins@land_100)) & isnull(mask_ice_basin), 1, null())"
r.mapcalc "mask_ice_land_basin = if(not(isnull(basins_filled@land_100)), 1, null())"

g.region RACMO@RACMO -p
FILE=${DATADIR}/RACMO/freshwater/Icemask_Topo_Iceclasses_lon_lat_average_1km.nc
r.in.gdal -o input="NetCDF:${FILE}:Promicemask" output=mask_ice
r.region map=mask_ice region=RACMO@RACMO
r.mapcalc "mask_ice_RACMO  = if((mask_ice != 0) & mask_ice_land_basin, 1, null())"
r.mapcalc "mask_land_RACMO = if((mask_ice == 0) & mask_ice_land_basin, 1, null())"
g.region -d

r.mapcalc "model = 1"
r.region map=model region=RACMO@RACMO

# basin_partition: extract one time step from a NetCDF and compute per-basin sum
function basin_partition() {
    JOB="$1"      # 0 to ndays-1
    SLOT="$2"     # 0 to ncpus-1
    ZONES="$3"    # e.g. basins@land_100
    RCMfilevar=$4 # e.g. RACMO_runoff_ice_2000.nc:runoff
    REGION=$5     # e.g. RACMO@RACMO

    RCMfile=${RCMfilevar/:*/}

    r.external -o source="netCDF:tmp/${RCMfilevar}" output=model_${SLOT} band=${JOB} --o --q 2>/dev/null
    r.region map=model_${SLOT} region=${REGION} --q
    DATESTR0=$(ncdump -h tmp/${RCMfile} | grep time:units)
    DATESTR0=${DATESTR0: 27:10}
    DATESTR=$(date -d"${DATESTR0}+$(( ${JOB}-1 )) days" +"%Y-%m-%d")
    r.univar -t map=model_${SLOT} zones=${ZONES} --q 2>/dev/null | cut -d"|" -f13 | datamash -t"|" transpose | sed s/^sum/${DATESTR}/
}
export -f basin_partition

g.mapset RACMO
log_info "Using EXTRA GNU parallel options via \${PARALLEL}: ${PARALLEL:-Not set}"

# RACMO ice runoff (1958-2024)
k_pct=100
for y in $(seq 1958 2024); do
    file_base=RACMO_runoff_ice_${y}
    infile=${file_base}.nc
    outfile=${file_base}_${k_pct}.bsv

    log_info "RACMO ice: ${outfile}"

    if [[ -e ./dat/${outfile} ]]; then
        log_warn "${outfile} exists. Skipping..."
    else
        cp ./tmp/header_ice_${k_pct}.bsv ./dat/${outfile}
        NT=$(ncdump -h ./tmp/${infile} | sed -n 's/.*time = \([0-9]\+\) ;/\1/p')
        seq 0 $((NT - 1)) \
            | parallel --nice 9 --keep-order --bar \
                basin_partition {#} {%} basins@ice_${k_pct} "${infile}:runoff" RACMO@RACMO \
            >> ./dat/${outfile}
    fi
done

# RACMO land runoff (1958-2024)
k_pct=100
for y in $(seq 1958 2024); do
    file_base=RACMO_runoff_land_${y}
    infile=${file_base}.nc
    outfile=${file_base}_${k_pct}.bsv

    log_info "RACMO land: ${outfile}"

    if [[ -e ./dat/${outfile} ]]; then
        log_warn "${outfile} exists. Skipping..."
    else
        cp ./tmp/header_land_${k_pct}.bsv ./dat/${outfile}
        NT=$(ncdump -h ./tmp/${infile} | sed -n 's/.*time = \([0-9]\+\) ;/\1/p')
        seq 0 $((NT - 1)) \
            | parallel --nice 9 --keep-order --bar \
                basin_partition {#} {%} basins@land_${k_pct} "${infile}:runoff" RACMO@RACMO \
            >> ./dat/${outfile}
    fi
done
