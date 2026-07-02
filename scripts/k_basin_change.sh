#!/usr/bin/env bash
# Generate basin difference maps showing how outlet routing changes across k values.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

set -x

g.mapset -c k_basin_change

log_info "Exporting outlet coordinates"

mapsets=$(g.mapsets -l separator="\n" | grep -E "^ice_*")
parallel --bar "r.out.xyz input=outlets@{1} | awk -F'|' '{print \$3, \$1}' > ./tmp/outlets_{1}_x" ::: ${mapsets}
parallel --bar "r.out.xyz input=outlets@{1} | awk -F'|' '{print \$3, \$2}' > ./tmp/outlets_{1}_y" ::: ${mapsets}

# Encode outlet (x,y) coordinates as basin category values, then create
# rasters where each cell stores the outlet coordinate of its basin.
log_info "Encoding outlet coordinates"

coord="x y"
parallel --bar "g.copy basins@{1},cat_{1}_{2}" ::: ${mapsets} ::: ${coord}
parallel --bar "r.category map=cat_{1}_{2} rules=./tmp/outlets_{1}_{2} separator=space" ::: ${mapsets} ::: ${coord}
parallel --bar "r.mapcalc '{1}_{2} = @cat_{1}_{2}'" ::: ${mapsets} ::: ${coord}

log_info "Exporting cells..."
log_warn "Reducing resolution."

LIST=$(g.list type=raster pattern='^[i|l][ceand]*_*[x|y]$' separator=,)
echo "x,y,${LIST}" | tr "," "|" > ./tmp/cells_xy.bsv
g.region res=1000 -a
r.out.xyz input=${LIST} output=- >> ./tmp/cells_xy.bsv
g.region -d

LIST=ice_100_x,ice_100_y,ice_80_x,ice_80_y,ice_90_x,ice_90_y,z_s@ArcticDEM
echo "x,y,${LIST}" | tr "," "|" > ./tmp/cells_elev.bsv
g.region res=1000 -a
r.out.xyz input=${LIST} output=- >> ./tmp/cells_elev.bsv
g.region -d

log_info "Calculating cell outlet statistics..."
"${PYTHON}" ./scripts/k_basin_change.py

log_info "Re-importing rasters."
log_warn "Reduced resolution."

g.region res=1000 -a
r.in.xyz input=./tmp/cells_xy_ice_stats.bsv output=ice_max method=mean x=4 y=5 z=3 skip=1
r.in.xyz input=./tmp/cells_xy_ice_stats.bsv output=ice_avg method=mean x=4 y=5 z=2 skip=1

# for raster in ice_max ice_avg land_max land_avg; do
for raster in ice_max ice_avg; do
  r.mask raster=mask_o_l_i@ArcticDEM maskcats="2 3" --o
  r.mapcalc "${raster}_land = if(isnull(${raster}), -1 * (mask_o_l_i@ArcticDEM >=2), ${raster})" --o
  r.mask -r
  r.reclass input=${raster}_land output=${raster}_class --o rules=./tmp/categories.txt

  # https://colorbrewer2.org/?type=sequential&scheme=YlGnBu&n=5
  cat << EOF | r.colors map=${raster}_class rules=-
-1 128:128:128
5 255:255:204
4 161:218:180
3 65:182:196
2 44:127:184
1 37:52:148
EOF
done

# Coastline vectors
g.mapset k_basin_change

g.region -d
r.mapcalc "outline = if(mask_o_l_i@ArcticDEM != 1)"
r.to.vect input=outline output=outline type=area

g.region res=1000 -a
r.mapcalc "outline_low = if(mask_o_l_i@ArcticDEM != 1, 1, null())"
r.to.vect input=outline_low output=outline_low type=area
g.region -d

# Elevation contours
r.mask outline_low maskcats=1 --o
g.region res=1000 -a
r.mapcalc "z_s = z_s@BedMachine"
r.contour input=z_s output=z_s levels=1000,1500,2000
g.region -d
r.mask -r

# Generate PNG figures
log_info "Generating graphic..."
mkdir -p ./fig

for domain in ice; do
  for stat in max avg; do
    raster=${domain}_${stat}_class

    g.region -d
    g.region res=1000 -a
    r.mask -r

    cat << EOF | ps.map input=- output=./tmp/${raster}.ps --o
    border n
    paper a3
      end

    raster ${raster}

    vlines z_s
      type line
      where level = 1500
      color black
      width 1
      end
    vlines z_s
      type line
      where level = 1500
      color white
      width 5
      end

    colortable y
      nodata n
      where 5.5 12
      fontsize 24
      end
    scalebar s
      length 100
      units kilometers
      segment 1
      where 7 11.5
      fontsize 24
      end

EOF

    convert -trim -transparent white ./tmp/${raster}.ps ./fig/${raster}.png
    g.region -d
  done
done
