# CARRA2 Integration Plan

Add CARRA2 (Copernicus pan-Arctic Regional Reanalysis, 2nd generation) as a third RCM
alongside MAR and RACMO.

## Key constraint

CARRA2 **does not simulate glacial ice melt** (NWP surface model limitation). The user
guide states explicitly: *"over glaciers this runoff estimate will in many cases be low
and unrealistic"*. Therefore CARRA2 can only be used for the **land (tundra) domain** —
it cannot replace MAR/RACMO for the ice domain.

## Dataset facts

| Property | Value |
|---|---|
| Variable | `surface_runoff`, GRIB short name `surfro`, cfgrib name likely `sro` |
| Units | kg m⁻² (= mm) — accumulated per forecast, de-accumulated to daily |
| Resolution | 2.5 km polar stereographic, 2880 × 2880 grid |
| Temporal coverage | 1986 – present |
| CDS dataset (daily) | `reanalysis-pan-carra-means`, `product_type: daily_accumulation` |
| CDS dataset (sub-daily) | `reanalysis-pan-carra` |
| Format | GRIB2 recommended (NetCDF still experimental) |
| Projection | Polar stereographic — **no EPSG documented**; extract params from GRIB |

## New and changed files

| File | Action |
|---|---|
| `scripts/carra_download.py` | **New** — CDS API download loop, one GRIB per year |
| `scripts/rcm_preprocess.py` | Add CARRA block: GRIB → NC + area correction |
| `scripts/area_error.sh` | Add CARRA mapset + grid registration in GRASS |
| `scripts/rcm.sh` | Add CARRA land-only basin partitioning loop (1986–present) |
| `Makefile` | Add `carra_download.py` to `stamps/RCM` deps |
| `scripts/bsv2netcdf.py` | No changes needed |
| `scripts/coverage_calc.sh` | Add `coverage_CARRA_land_100` stats block |
| `scripts/coverage_csv2nc.py` | Automatically covers CARRA (loops over all models) |

## Implementation steps

### Step 1 — Download (`scripts/carra_download.py`)

```python
import cdsapi, os

c = cdsapi.Client()   # requires ~/.cdsapirc with CDS credentials

for year in range(1986, 2026):
    outfile = f"./tmp/CARRA_runoff_land_{year}.grib"
    if os.path.exists(outfile):
        continue
    c.retrieve(
        'reanalysis-pan-carra-means',
        {
            'variable': 'surface_runoff',
            'product_type': 'daily_accumulation',
            'year': str(year),
            'month': [f'{m:02d}' for m in range(1, 13)],
            'day':   [f'{d:02d}' for d in range(1, 32)],
            'format': 'grib',
            'area': [84, -74, 59, -11],   # N, W, S, E — Greenland bounding box
        },
        outfile
    )
```

Requires `pip install cdsapi` (already in `environment-dev.yml` presumably, otherwise add it).

### Step 2 — Verify GRIB structure (do this before writing preprocessing)

```bash
python -c "
import cdsapi
c = cdsapi.Client()
c.retrieve('reanalysis-pan-carra-means',
    {'variable': 'surface_runoff', 'product_type': 'daily_accumulation',
     'year': '2000', 'month': '07', 'day': ['01','02','03'],
     'format': 'grib', 'area': [84, -74, 59, -11]},
    'test_carra.grib')
"
python -c "import xarray as xr; print(xr.open_dataset('test_carra.grib', engine='cfgrib'))"
```

Use the output to confirm:
- Exact variable name in cfgrib (expected `sro`, may differ)
- Time encoding (check daily totals are truly per-day, not cumulative from year start)
- Actual projection parameters (needed for GRASS import)

Extract projection parameters from the GRIB file:
```bash
grib_get -p latitudeOfFirstGridPointInDegrees,longitudeOfFirstGridPointInDegrees,\
Nx,Ny,DxInMetres,DyInMetres,LaDInDegrees,orientationOfTheGridInDegrees \
test_carra.grib
```

### Step 3 — Area error correction (add to `scripts/area_error.sh`)

Add a CARRA mapset section after the RACMO block. Use the grid extents extracted from
the GRIB in Step 2 to set the GRASS region, then apply the same `v.mkgrid` → PROJ
distortion → `r.in.xyz` pattern already used for MAR and RACMO:

```bash
log_info "Creating CARRA mapset..."
g.mapset -c CARRA
# set region from GRIB-derived extents, e.g.:
# g.region w=<x0> e=<x1> s=<y0> n=<y1> res=2500 -p
# g.region save=CARRA
# ... same area-error block as MAR/RACMO ...
```

### Step 4 — Preprocess (add to `scripts/rcm_preprocess.py`)

```python
# CARRA land runoff — land domain only
import cfgrib   # pip install cfgrib

area = xr.open_dataset("./tmp/err_2D_area_CARRA.nc")

for year in range(1986, 2026):
    outfile = f"./tmp/CARRA_runoff_land_{year}.nc"
    if os.path.exists(outfile):
        continue
    infile = f"./tmp/CARRA_runoff_land_{year}.grib"
    print("infile:", infile, "outfile:", outfile)
    ds = xr.open_dataset(infile, engine="cfgrib")
    # variable name — verify from Step 2, likely 'sro'
    da = ds['sro'] * area['Band1'].values
    da.variable.attrs = {'units': 'kg m-2 day-1',
                         'long_name': 'Surface runoff',
                         'standard_name': 'surface_runoff_flux'}
    ds_sub = da.to_dataset(name='runoff')
    ds_sub.to_netcdf(outfile)
```

### Step 5 — Basin partitioning (add to `scripts/rcm.sh`)

Add after the RACMO block. Land domain only, k=100 only:

```bash
# CARRA land runoff setup
g.mapset -c CARRA
g.region -d
r.mapcalc "mask_land_basin = if(not(isnull(basins@land_100)) & isnull(mask_ice_basin), 1, null())"
# Use land cells from the existing ice/land/ocean mask (no dedicated tundra mask in CARRA)
r.mapcalc "mask_land_CARRA = if(mask_o_l_i@ArcticDEM == 2, 1, null())"
g.region -d

# Basin partitioning loop
g.mapset CARRA
k_pct=100
for y in $(seq 1986 2025); do
  file_base=CARRA_runoff_land_${y}
  infile=${file_base}.nc
  outfile=${file_base}_${k_pct}.bsv

  log_info "Calculating CARRA land runoff: ${outfile}"

  if [[ -e ./dat/${outfile} ]]; then
    log_warn "Output file ${outfile} exists. Skipping..."
  else
    cp ./tmp/header_land_${k_pct}.bsv ./dat/${outfile}
    NT=$(ncdump -h ./tmp/${infile} | sed -n 's/.*time = \([0-9]\+\) ;/\1/p')
    seq 0 $((NT - 1)) \
      | parallel --nice 9 --keep-order --bar basin_partition {#} {%} \
          basins@land_${k_pct} "${infile}:runoff" CARRA@CARRA \
          >> ./dat/${outfile}
  fi
done
```

### Step 6 — Makefile

```makefile
stamps/RCM: $(SCRIPTS)/carra_download.py $(SCRIPTS)/rcm_preprocess.py $(SCRIPTS)/rcm.sh
    mkdir -p dat
    python $(SCRIPTS)/carra_download.py
    python $(SCRIPTS)/rcm_preprocess.py
    grass ./G/PERMANENT --exec $(SCRIPTS)/rcm.sh
    @mkdir -p stamps && touch $@
```

Also add `cdsapi` and `cfgrib` to `environment.yml`.

## Open questions before implementation

1. Confirm cfgrib variable name for `surfro` (Step 2 test download)
2. Confirm CARRA2 projection parameters (needed for `area_error.sh` and GRASS region)
3. Decide whether to include CARRA2 `discharge.py` output — discharge.py will automatically
   pick up the new NetCDF files if they are placed in `freshwater/land_100/discharge/`,
   so no code change needed there
4. Check if `daily_accumulation` product type truly gives one value per calendar day
   (some CDS products use the term differently)

## References

- [CARRA2 Data User Guide](https://confluence.ecmwf.int/display/CKB/Copernicus+pan-Arctic+Regional+Reanalysis+%28CARRA2%29%3A+Data+User+Guide)
- [CDS dataset — daily/monthly](https://cds.climate.copernicus.eu/datasets/reanalysis-pan-carra-means)
- [CDS dataset — sub-daily](https://cds.climate.copernicus.eu/datasets/reanalysis-pan-carra)
