#!/usr/bin/env bash
# Import all external datasets into GRASS GIS mapsets.

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

# GIMP Ocean Mask
g.mapset -c GIMP_0714
r.in.gdal -o input=${DATADIR}/MEaSUREs/NSIDC-0714.001/GimpOceanMask_90m_2000_v1.2.tif output=mask_ocean_lakes
# ocean mask has some cut-off oceans a.k.a lakes. See e.g. near -274731,-2747470.
g.region raster=mask_ocean_lakes

# Fill the lakes and treat as land.
r.null map=mask_ocean_lakes setnull=0
r.clump input=mask_ocean_lakes output=clumps
main_clump=$(r.stats -c -n clumps sort=desc | head -n1 | cut -d" " -f1)
r.mapcalc "mask_ocean = if(clumps == ${main_clump}, 1, null())"
r.null mask_ocean null=0

# Citterio 2013 Ice Mask
g.mapset -c Citterio_2013

mkdir -p tmp

# fix bad shapefile that doesn't import directly into GRASS.
ogr2ogr -s_srs EPSG:4326 -t_srs EPSG:3413 ./tmp/PROMICE_3413 ${DATADIR}/Citterio_2013/PROMICE_250_2019-12-04_EPSG4326
v.in.ogr input=./tmp/PROMICE_3413 output=mask

g.region raster=mask_ocean@GIMP_0714
v.to.rast input=mask output=mask_ice_holes use=val val=1

# Fill nunataks so everything routes to the coast (no interior drainage basins).
r.mapcalc "not_ice = if(isnull(mask_ice_holes), 1, null())"
r.clump input=not_ice output=clumps
main_clump=$(r.stats -c -n clumps sort=desc | head -n1 | cut -d" " -f1) # ocean
r.mapcalc "mask_ice = mask_ice_holes ||| if(clumps != ${main_clump})"
r.null mask_ice null=0

# ArcticDEM
log_info "Importing ArcticDEM"
g.mapset -c ArcticDEM

r.external source=${DATADIR}/ArcticDEM/arcticdem_mosaic_100m_v4.1_dem.tif output=arctic_raw

g.region raster=mask_ocean@GIMP_0714 align=arctic_raw
g.region save=ArcticDEM
g.mapset PERMANENT
g.region region=ArcticDEM@ArcticDEM
g.region -sp
g.mapset ArcticDEM

r.mapcalc "mask_land = if((mask_ice@Citterio_2013 == 0) && (mask_ocean@GIMP_0714 == 0), 1, 0)"

# set up mask: (o)cean=1 (l)and=2 (i)ce=3
r.mapcalc "mask_o_l_i_4 = (mask_ocean@GIMP_0714 + mask_land*2 + mask_ice@Citterio_2013*3)"
r.mapcalc "mask_o_l_i = if(mask_o_l_i_4 == 4, 1, mask_o_l_i_4)" # cleanup multi-flagged ocean cells
r.mapcalc "z_s_ocean = arctic_raw"
r.null map=z_s_ocean null=0
r.mapcalc "z_s = if(mask_o_l_i == 1, null(), z_s_ocean)"

# BedMachine v5
log_info "Importing BedMachine"
g.mapset -c BedMachine

for var in mask surface thickness bed errbed geoid source; do
  echo $var
  r.in.gdal -o input="NetCDF:${DATADIR}/Morlighem_2017/BedMachineGreenland-v5.nc:"${var} output=${var}
done
g.rename raster=surface,z_s
g.rename raster=bed,z_b

g.region raster=z_s
g.region res=150 -pa

# NSIDC 0713 Basemap
log_info "Importing NSIDC 0713"
g.mapset -c NSIDC_0713
g.region -d
r.external -o input=${DATADIR}/GIMP/0713/RGB.vrt output=rgb

# Mouginot 2019
log_info "Loading Mouginot 2019"
g.mapset -c Mouginot_2019
v.import input=${DATADIR}/Mouginot_2019/Greenland_Basins_PS_v1.4.2.shp output=basins snap=1

# Zwally 2012
log_info "Loading Zwally 2012"
g.mapset -c Zwally_2012
v.import input=${DATADIR}/Zwally_2012/sectors output=sectors snap=1

# Mankoff 2020 gates
g.mapset -c Mankoff_2020
log_info "Importing Mankoff 2020 gates"

mkdir -p tmp/Mankoff_2020_gates

dest=./tmp/Mankoff_2020_gates/gates.gpkg
if [[ ! -f ${dest} ]]; then
  wget "https://dataverse.geus.dk/api/access/datafile/:persistentId?persistentId=doi:10.22008/promice/data/ice_discharge/gates/v02/UFW2WY" -O ${dest}
fi

md5sum ./tmp/Mankoff_2020_gates/*
# expected: d28f4a1c4c72490dc83893c18352f579  ./tmp/Mankoff_2020_gates/gates.gpkg

v.in.ogr input=./tmp/Mankoff_2020_gates/gates.gpkg output=gates
v.db.select map=gates|head

# Bjørk 2015 names
g.mapset -c Bjork_2015
log_info "Importing Bjørk 2015 names"
v.in.ogr input=${DATADIR}/Bjørk_2015/EPSG_3413/ output=names
v.db.select map=names|head

# x and y rasters in PERMANENT mapset
log_info "x, y, ones..."
MAPSET=$(g.mapset -p)
g.mapset PERMANENT
r.mapcalc "x = x()"
r.mapcalc "y = y()"
r.mapcalc "ones = 1"
g.mapset ${MAPSET}

# Bamber 2018 (used for Disko island figure outlines)
g.mapset -c Bamber_2018
FILE=${DATADIR}/Bamber_2018/FWF17.v3_a.nc

cdo -v -b F32 timsum ${FILE} ./tmp/B2018.nc
r.external source=netcdf:./tmp/B2018.nc:runoff_ice output=ice
r.external source=netcdf:./tmp/B2018.nc:runoff_tundra output=tundra

r.mask -r
r.mapcalc "MASK = if(ice == 0, null(), 1)" --o
r.to.vect input=ice output=ice type=area
r.mask -r
r.mapcalc "MASK = if(tundra == 0, null(), 1)" --o
r.to.vect input=tundra output=tundra type=area
r.mask -r
