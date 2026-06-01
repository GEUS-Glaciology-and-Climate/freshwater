# Greenland liquid water discharge from 1950 through 2024

This is the source for "Greenland liquid water discharge from 1958 through 2019" and subsequent versions.

- Paper: https://doi.org/10.5194/essd-12-2811-2020
- Data: https://doi.org/10.22008/promice/freshwater
- Companion paper (solid ice discharge): https://doi.org/10.5194/essd-12-1367-2020
  - Source: https://github.com/GEUS-PROMICE/ice_discharge/
  - Data: https://doi.org/10.22008/promice/data/ice_discharge

Source hosted on GitHub at https://github.com/GEUS-PROMICE/freshwater.
GitHub [issues](https://github.com/mankoff/freshwater/issues) are used to collect suggested improvements or problems.

## Updates since last publication

### v 2025

[release_2025](https://github.com/GEUS-Glaciology-and-Climate/freshwater/tree/release_2025) changes from v2023:

- Updated MAR data from 3.13 to 3.14
- Added MAR 2023 and 2024
- Added MAR 1940 through 1958
- Added RACMO 2023 and 2024
- One file per year per RCM
- Added `year` option to `discharge.py` API for faster queries

### v 2023

[release_2023](https://github.com/GEUS-Glaciology-and-Climate/freshwater/tree/release_2023) changes from v2022-08:

- Added 2022 RACMO and MAR data
- Updated MAR from 3.12 to 3.13
- Updated BedMachine from v4 to v5
- Updated ArcticDEM from v3.0 to v4.1

### v 2022-10

- No change to data values
- Reformatted to 4 NetCDF files (one per RCM × domain) each containing all years
- Discharge data moved into the Streams/Outlets/Basins dataset: https://doi.org/10.22008/FK2/XKQVL7

### v 2022-08

- Updated BedMachine from v3 to v4
- Data now spans 1950–2021 (was 1958–2019)
- Internal NetCDF variable renamed from `runoff` to `discharge`
- Significant metadata improvements supporting queries by basin, region, or sector

## WARNING

> Bugs may exist in this data or the `discharge.py` access script. All known bugs are documented at https://github.com/GEUS-PROMICE/freshwater/issues. Before using this software or finalizing results, check for open issues or recently closed ones.

## Related Work

- "Greenland ice sheet mass balance from 1840 through next week": https://doi.org/10.5194/essd-13-5001-2021
  - Source: https://github.com/GEUS-Glaciology-and-Climate/mass_balance

## Citation

```bibtex
@article{mankoff_2020_liquid,
  author    = {Mankoff, Kenneth D. and Noël, Brice and Fettweis, Xavier and Ahlstrøm, Andreas P. and
                  Colgan, William and Kondo, Ken and Langley, Kirsty and Sugiyama, Shin and van As,
                  Dirk and Fausto, Robert S.},
  title     = {{G}reenland liquid water discharge from 1958 through 2019},
  journal   = {Earth System Science Data},
  year      = 2020,
  volume    = 12,
  number    = 4,
  pages     = {2811--2841},
  DOI       = {10.5194/essd-12-2811-2020},
  publisher = {Copernicus GmbH}
}
```

## Funding

| Dates        | Organization | Program                            | Effort                                 |
|--------------|--------------|------------------------------------|----------------------------------------|
| 2023 –       | NASA GISS    | Modeling Analysis and Prediction   | Maintenance                            |
| 2022 –       | GEUS         | PROMICE                            | Distribution (data hosting)            |
| 2018 – 2022  | GEUS         | PROMICE                            | Development; publication; distribution |

## Accessing this data

### Introduction

Data can be accessed directly from the NetCDF files, which allows metadata-based queries such as _"all outlets within 5 km of Jakobshavn Isbræ"_.

The `discharge.py` script provides a higher-level interface for querying outlets, basins, and discharge by region of interest (ROI). The ROI can be a point, polygon coordinates, or a file (e.g. KML). The script works from the command line (CSV to stdout) or as a Python API (returns GeoPandas/xarray objects).

### Database layout

The `freshwater/` folder must contain `land/` and `ice/` subfolders, each with:

```
freshwater/
  ice/
    basins.csv, basins.gpkg, basins_filled.gpkg
    outlets.csv, outlets.gpkg
    streams.csv, streams.gpkg
    discharge/
      MAR_1940.nc ... MAR_2024.nc
      RACMO_1958.nc ... RACMO_2024.nc
  land/
    (same structure)
```

### Requirements

```bash
mamba create -n freshwater_user python=3.13 xarray=2025.1.2 fiona=1.10.1 shapely=2.0.7 \
    geopandas=1.0.1 netcdf4=1.7.2 dask=2025.2.0 pandas=2.2.3
mamba activate freshwater_user
```

Or use the `environment.yml` file in this repository.

### Command line examples

```bash
# Show usage
python discharge.py -h

# Outlets near Watson River (one land point)
python discharge.py --base ./freshwater --roi=-50.5,67.2 -o -q

# Discharge at Watson River for 2012
python discharge.py --base ./freshwater --roi=-50.5,67.2 --year=2012 -q -d | grep -E 'time|^2012-06-0'

# Polygon covering multiple outlets, including upstream ice
python discharge.py --base ./freshwater \
    --roi="-51.50,66.93 -51.21,66.74 -49.44,66.91 -49.84,67.18" -q -u -o
```

### Python API examples

```python
from discharge import discharge

# Outlets at a point
df = discharge(base="./freshwater", roi="-50.5,67.2", quiet=True).outlets()
df.drop(columns=["outlet", "basin"])

# Discharge for a polygon, 2012 only
ds = discharge(base="./freshwater",
               roi="-51.50,66.93 -51.21,66.74 -49.44,66.91 -49.84,67.18",
               year=2012, quiet=False, upstream=True).discharge()

# Sum outlets, apply rolling mean, plot
import matplotlib.pyplot as plt
dims = [d for d in ds.dims if d != 'time']
ds.sum(dim=dims).rolling(time=5).mean()\
  .sel(time=slice('2012-04-01', '2012-11-15'))\
  .plot()
plt.legend()
```

### Notes

- The script takes a few seconds for outlet queries and ~10–60 s per discharge dataset. With 6 possible datasets (2 RCMs × 3 domains), allow several minutes on a fast laptop.
- If a polygon includes ice outlets and `--upstream` is set, some outlets may appear twice (once direct, once upstream). Use pandas to deduplicate.
- `id` values are unique within a domain (`land` or `ice`) but not between them.
- Use `--roi=-60,60` syntax (not `--roi -60,60`) due to bash argument parsing.
- Longitude is degrees East (negative for Greenland).
