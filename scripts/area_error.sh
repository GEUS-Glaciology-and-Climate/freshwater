#!/usr/bin/env bash
# Compute 2D area correction rasters for EPSG:3413 for MAR and RACMO grids.
#
# Generates ./tmp/err_2D_area_MAR.nc and ./tmp/err_2D_area_RACMO.nc.
# Applied later in rcm_preprocess.py to correct for grid-cell area distortion.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

set -x
log_info "Area Error..."

# MAR grid
# MAR on ISMIP6 grid: lower-left at (-720000m, -3450000m), 1km cells
log_info "Creating MAR mapset..."
g.mapset -c MAR

g.region w=-720000 e=960000 s=-3450000 n=-570000 res=1000 -pa
g.region w=w-500 e=e+500 n=n+500 s=s-500 res=1000 -p
g.region save=MAR

if [[ "" == $(g.list type=raster pattern=err_2D_area mapset=.) ]]; then
  MAPSET=$(g.mapset -p)
  log_info "2D Area Error for ${MAPSET}"

  v.mkgrid map=grid position=region type=point
  v.out.ascii grid | m.proj input=- -od | cut -d"|" -f1,2 | tr '|' ' ' > ./tmp/distortion_ll_${MAPSET}.txt
  PROJSTR=$(g.proj -j | grep -v 'crs')
  echo $PROJSTR
  cat ./tmp/distortion_ll_${MAPSET}.txt \
    | proj -VS ${PROJSTR} \
    | grep Areal \
    | column -t \
    | sed s/\ \ /,/g \
    | cut -d, -f4 \
      > ./tmp/distortion_err_${MAPSET}.txt

  paste -d " " <(m.proj -i input=./tmp/distortion_ll_${MAPSET}.txt separator=space | cut -d" " -f1,2) ./tmp/distortion_err_${MAPSET}.txt | r.in.xyz input=- output=err_2D_inv separator=space

  r.mapcalc "err_2D_area = 1/(err_2D_inv)"
  r.mapcalc "err_2D_line = 1/(err_2D_inv^0.5)"
  r.out.gdal --o -c -m input=err_2D_area output=./tmp/err_2D_area_${MAPSET}.nc type=Float32 format=netCDF
fi

# RACMO grid
log_info "Creating RACMO mapset..."
g.mapset -c RACMO
FILE=${DATADIR}/RACMO/freshwater/Icemask_Topo_Iceclasses_lon_lat_average_1km.nc
cdo -sinfo ${FILE}
x0=$(ncdump -v x ${FILE} | grep "^ x =" | cut -d" " -f4 | cut -d, -f1)
x1=$(ncdump -v x ${FILE} | tail -n2 | head -n1 | tr ',' '\n' | tail -n1 | cut -d" " -f2)
y0=$(ncdump -v y ${FILE} | grep "^ y =" | cut -d" " -f4 | cut -d, -f1)
y1=$(ncdump -v y ${FILE} | tail -n2 | head -n1 | tr ',' '\n' | tail -n1 | cut -d" " -f2)
g.region w=$x0 e=$x1 s=$y0 n=$y1 res=1000 -p
g.region s=s-500 n=n+500 e=e+500 w=w-500 -p
g.region save=RACMO

if [[ "" == $(g.list type=raster pattern=err_2D_area mapset=.) ]]; then
  MAPSET=$(g.mapset -p)
  log_info "2D Area Error for ${MAPSET}"

  v.mkgrid map=grid position=region type=point
  v.out.ascii grid | m.proj input=- -od | cut -d"|" -f1,2 | tr '|' ' ' > ./tmp/distortion_ll_${MAPSET}.txt
  PROJSTR=$(g.proj -j | grep -v 'crs')
  echo $PROJSTR
  cat ./tmp/distortion_ll_${MAPSET}.txt \
    | proj -VS ${PROJSTR} \
    | grep Areal \
    | column -t \
    | sed s/\ \ /,/g \
    | cut -d, -f4 \
      > ./tmp/distortion_err_${MAPSET}.txt

  paste -d " " <(m.proj -i input=./tmp/distortion_ll_${MAPSET}.txt separator=space | cut -d" " -f1,2) ./tmp/distortion_err_${MAPSET}.txt | r.in.xyz input=- output=err_2D_inv separator=space

  r.mapcalc "err_2D_area = 1/(err_2D_inv)"
  r.mapcalc "err_2D_line = 1/(err_2D_inv^0.5)"
  r.out.gdal --o -c -m input=err_2D_area output=./tmp/err_2D_area_${MAPSET}.nc type=Float32 format=netCDF
fi
