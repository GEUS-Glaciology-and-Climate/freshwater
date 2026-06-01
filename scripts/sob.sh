#!/usr/bin/env bash
# Calculate streams, outlets, and basins (SOB) for ice and land domains.
#
# Runs routing for:
#   - land domain (k=1.0 only)
#   - ice domain (k=0.8, 0.9, 1.0)
# Land is run first because ice domains need the downstream land outlet.

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

for domain in land ice; do
  for k_pct in 100 90 80; do

    # land domain only uses k=100
    [[ ${domain} == land ]] && [[ ${k_pct} != 100 ]] && break

    k=$(echo "scale=2; ${k_pct}/100" | bc -l)

    log_info "Setting domain to ${domain}_${k_pct}"
    g.mapset -c ${domain}_${k_pct}
    g.region -dp

    log_info "Masking domain to ${domain}"
    [[ ${domain} == "ice" ]]  && r.mask raster=mask_o_l_i@ArcticDEM maskcats=3 --o
    [[ ${domain} == "land" ]] && r.mask raster=mask_o_l_i@ArcticDEM maskcats="2 3" --o

    # remove islands (value is in hectares)
    # frink "1 km^2 -> hectares" # 100
    if [[ ${domain} == "land" ]]; then
      r.reclass.area -d input=MASK output=MASK_nosmall value=100.1 mode=lesser method=rmarea
      r.mask MASK_nosmall --o
    fi

    log_info "Calculating subglacial head with k: ${k} (${domain})"
    r.mapcalc "head = if(mask_o_l_i@ArcticDEM == 1, null(), 0) + if(mask_o_l_i@ArcticDEM == 2, z_s@ArcticDEM, 0) + if(mask_o_l_i@ArcticDEM == 3, (z_s@ArcticDEM - thickness@BedMachine) + ${k} * 0.917 * thickness@BedMachine)"
    THRESH=300
    log_warn "Using threshold: ${THRESH}"
    log_info "r.stream.extract..."

    r.stream.extract elevation=head threshold=${THRESH} memory=16384 direction=dir stream_raster=streams stream_vector=streams
    log_info "Calculating outlets"
    r.mapcalc "outlets_1 = if(dir < 0, 1, null())"
    r.out.xyz input=outlets_1 | \
        cat -n | \
        tr '\t' '|' | \
        cut -d"|" -f1-3 | \
        v.in.ascii input=- output=outlets_uniq separator=pipe \
            columns="x int, y int, cat int" x=2 y=3 cat=1
    log_info "r.stream.basins..."
    g.extension extension=r.stream.basins

    r.stream.basins -m direction=dir points=outlets_uniq basins=basins_uniq memory=16384 --verbose

    # absorb small basins and outlets (minsize is in cells)
    # frink "1.0 km^2 / ((100 * 100) m^2)" # 100
    r.clump -d input=basins_uniq output=basins_nosmall minsize=101
    r.mode base=basins_nosmall cover=basins_uniq output=basins
    r.to.vect -v input=basins output=basins type=area
    v.db.dropcolumn map=basins column="label"

    v.to.rast input=outlets_uniq output=outlets_uniq use=cat
    r.mapcalc "outlets = if(outlets_uniq == basins, basins, null())"
    r.to.vect -v input=outlets output=outlets type=point
    v.db.dropcolumn map=outlets column=label

    if [[ ${domain} == "ice" ]]; then
      log_info "Adjusting ice basins..."
      g.rename raster=basins,basins_filled vector=basins,basins_filled
      r.mapcalc "basins = basins_filled * mask_ice_holes@Citterio_2013"
      r.to.vect -v input=basins output=basins type=area
      v.db.dropcolumn map=basins column="label"
    fi

    if [[ ${domain} == "land" ]]; then
      log_info "Adjusting land basins..."
      g.rename raster=basins,basins_filled vector=basins,basins_filled
      r.mapcalc "basins = basins_filled * not_ice@Citterio_2013"
      r.to.vect -v input=basins output=basins type=area
      v.db.dropcolumn map=basins column="label"
    fi

    log_info "Adding metadata..."

    # streams
    v.to.db map=streams option=length column=length

    # outlets
    v.db.addcolumn map=outlets column="lon DOUBLE PRECISION"
    v.db.addcolumn map=outlets column="lat DOUBLE PRECISION"
    v.db.addcolumn map=outlets column="x INT"
    v.db.addcolumn map=outlets column="y INT"
    v.db.addcolumn map=outlets column="elev INT"

    v.what.rast map=outlets raster=x@PERMANENT column=x
    v.what.rast map=outlets raster=y@PERMANENT column=y
    v.what.rast map=outlets raster=z_b@BedMachine column=elev

    mkdir -p tmp
    db.select -c sql='select x,y,cat from outlets' | m.proj -od input=- | tr '|' ',' > ./tmp/lonlat.csv
    db.in.ogr input=./tmp/lonlat.csv output=lonlat
    db.select table=lonlat|head
    v.db.join map=outlets column=cat other_table=lonlat other_column=field_3
    v.db.update map=outlets column=lon query_column=field_1
    v.db.update map=outlets column=lat query_column=field_2
    v.db.dropcolumn map=outlets columns=field_1,field_2,field_3
    db.select table=outlets | head

    if [[ ${domain} == "ice" ]]; then
       ice_domain=$(g.mapset -p)
       land_domain=land_100

       v.db.addcolumn map=outlets column="coast_id int"
       v.what.rast map=outlets type=point raster=basins_filled@${land_domain} column=coast_id

       v.db.addcolumn map=outlets column="coast_lon double"
       v.db.addcolumn map=outlets column="coast_lat double"
       v.db.addcolumn map=outlets column="coast_x int"
       v.db.addcolumn map=outlets column="coast_y int"

       g.copy vector=outlets@${land_domain},oland
       db.execute sql='UPDATE outlets SET coast_lon=(SELECT lon from oland WHERE outlets.coast_id=oland.cat)'
       db.execute sql='UPDATE outlets SET coast_lat=(SELECT lat from oland WHERE outlets.coast_id=oland.cat)'
       db.execute sql='UPDATE outlets SET coast_x=(SELECT x from oland WHERE outlets.coast_id=oland.cat)'
       db.execute sql='UPDATE outlets SET coast_y=(SELECT y from oland WHERE outlets.coast_id=oland.cat)'
    fi

    # Nearest upstream Zwally sector or Mouginot basin/region for each hydro outlet
    v.db.addcolumn map=outlets columns="Z2012_sector INT"
    v.db.addcolumn map=outlets columns="Z2012_sector_dist INT"
    v.distance from=outlets to=sectors@Zwally_2012 upload=to_attr column=Z2012_sector to_column=cat_
    v.distance from=outlets to=sectors@Zwally_2012 upload=dist column=Z2012_sector_dist

    v.db.addcolumn map=outlets columns="M2019_ID INT"
    v.db.addcolumn map=outlets columns="M2019_ID_dist INT"
    v.db.addcolumn map=outlets columns="M2019_basin CHAR(99)"
    v.db.addcolumn map=outlets columns="M2019_region CHAR(99)"
    v.distance from=outlets to=basins@Mouginot_2019 upload=to_attr column=M2019_ID to_column=cat output=connection
    v.distance from=outlets to=basins@Mouginot_2019 upload=dist column=M2019_ID_dist
    v.distance from=outlets to=basins@Mouginot_2019 upload=to_attr column=M2019_basin to_column=NAME
    v.distance from=outlets to=basins@Mouginot_2019 upload=to_attr column=M2019_region to_column=SUBREGION1

    # Nearest Mankoff 2020 gate and distance to gate
    v.db.addcolumn map=outlets columns="M2020_gate INT"
    v.db.addcolumn map=outlets columns="M2020_gate_dist INT"
    v.distance from=outlets to=gates@Mankoff_2020 upload=to_attr column=M2020_gate to_column=gate
    v.distance from=outlets to=gates@Mankoff_2020 upload=dist column=M2020_gate_dist

    # Nearest Bjørk 2015 name and distance
    v.db.addcolumn map=outlets columns="B2015_name CHAR(99)"
    v.db.addcolumn map=outlets columns="B2015_dist INT"
    v.distance from=outlets to=names@Bjork_2015 upload=to_attr column=B2015_name to_column=Official_n
    v.distance from=outlets to=names@Bjork_2015 upload=dist column=B2015_dist

    # basins
    v.to.db map=basins option=area column=area
    log_info "r.stream.order: BEGIN"
    g.extension extension=r.stream.order
    date
    time r.stream.order -m stream_rast=streams direction=dir elevation=head accumulation=ones@PERMANENT stream_vect=stream_vect strahler=strahler horton=horton shreve=shreve hack=hack topo=topological memory=16384
    date
    log_info "r.stream.order: END"

    for c in $(echo strahler horton shreve hack drwal_old topo_dim); do
        db.execute sql="ALTER TABLE streams ADD COLUMN ${c} INT"
        db.execute sql="UPDATE streams SET ${c}=(SELECT ${c} from stream_vect WHERE stream_vect.cat=streams.cat)"
    done

    for c in $(echo stright sinosoid cum_length source_elev outlet_elev); do
        db.execute sql="ALTER TABLE streams ADD COLUMN ${c} double"
        db.execute sql="UPDATE streams SET ${c}=(SELECT ${c} from stream_vect WHERE stream_vect.cat=streams.cat)"
    done

    # fix typos: sinosoid -> sinusoid, stright -> straight
    db.execute sql="ALTER TABLE streams ADD COLUMN sinusoid DOUBLE"
    db.execute sql="UPDATE streams SET sinusoid = sinosoid"
    v.db.dropcolumn map=streams columns=sinosoid

    db.execute sql="ALTER TABLE streams ADD COLUMN straight DOUBLE"
    db.execute sql="UPDATE streams SET straight = stright"
    v.db.dropcolumn map=streams columns=stright

    log_info "Exporting..."
    MAPSET=$(g.mapset -p)
    mkdir -p freshwater/${MAPSET}

    parallel --bar "db.select table={} | tr '|' ',' > ./freshwater/${MAPSET}/{}.csv" ::: streams outlets basins
    parallel --bar "v.out.ogr input={} output=./freshwater/${MAPSET}/{}.gpkg --o" ::: streams outlets basins basins_filled
  done
done
