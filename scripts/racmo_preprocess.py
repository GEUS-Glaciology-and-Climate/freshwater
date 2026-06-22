#!/usr/bin/env python
# Extract RACMO ice and land runoff to per-year NetCDF files in ./tmp/.
#
# Ice runoff: provided at 1 km on EPSG:3413 by Brice Noël.
# Land runoff: originally at 5.5 km on a rotated-pole grid, regridded to 1 km
#              EPSG:3413 (see $DATADIR/RACMO/freshwater/runoff_land_1km_regrid/).
#
# Applies the 2D area correction (from area_error.sh). Already-existing output
# files are skipped.

import xarray as xr
import numpy as np
import os
import glob
from tqdm import tqdm
from dask.diagnostics import ProgressBar

DATADIR = os.environ['DATADIR']
ROOT = DATADIR + "/RACMO/freshwater"
area = xr.open_dataset("./tmp/err_2D_area_RACMO.nc")

# Ice runoff (variable runoffcorr, mask Promicemask != 0)
msk = xr.open_dataset(ROOT + "/Icemask_Topo_Iceclasses_lon_lat_average_1km.nc")
msk['landmask'] = (msk['Promicemask'] == 0) & (msk['Topography'] != 0)

filelist = np.sort(glob.glob(ROOT + "/runoff_ice_1km/runoff*.nc"))
years = np.unique([_.split(".")[1].split("_")[0] for _ in filelist])
for i, y in enumerate(tqdm(years)):
    outfile = "./tmp/RACMO_runoff_ice_" + y + ".nc"
    if os.path.exists(outfile):
        continue

    filelist_year = [_ for _ in filelist if str(y) in _]
    print("infile:", filelist_year, "outfile:", outfile)
    ds = xr.open_mfdataset(filelist_year, combine='by_coords')
    da = (ds['runoffcorr'] * (msk['Promicemask'] != 0) * area['Band1'].values)
    da.variable.attrs = {'units': 'mm w.e. per day',
                         'long_name': 'Downscaled corrected snowmelt',
                         'standard_name': 'Downscaled_corrected_snowmelt'}
    ds_sub = da.to_dataset(name='runoff')
    with ProgressBar():
        ds_sub.to_netcdf(outfile, format="NETCDF4", engine="netcdf4")
    del ds, da, ds_sub

# Land runoff (variable runoff, tundra mask)
ROOT_REGRID = ROOT + "/runoff_land_1km_regrid"
msk = xr.open_dataset(ROOT_REGRID + "/Tundra_Mask_1km.nc")

filelist = np.sort(glob.glob(ROOT_REGRID + "/out/runoff*.nc"))
years = np.unique([_.split(".")[1] for _ in filelist])
for i, y in enumerate(tqdm(years)):
    outfile = "./tmp/RACMO_runoff_land_" + y + ".nc"
    if os.path.exists(outfile):
        continue

    print("infile:", filelist[i], "outfile:", outfile)
    ds = xr.open_mfdataset(filelist[i], combine='by_coords')
    ds = ds.drop_vars(['lon', 'lat'])
    ds = ds.squeeze()  # drop height=1

    da = ds['runoff'] * msk['Tundra'].values * area['Band1'].values
    da.variable.attrs = {'units': 'mm w.e. per day',
                         'long_name': 'Downscaled corrected snowmelt',
                         'standard_name': 'Downscaled_corrected_snowmelt'}
    ds_sub = da.to_dataset(name='runoff')

    # RACMO land time is 'days since 1950-01-01' across all files; re-encode per year.
    this_year = ds['time'].dt.year.values[0]
    ds_sub['time'].encoding['units'] = f'days since {this_year}-01-01'

    with ProgressBar():
        ds_sub.to_netcdf(outfile, encoding={'runoff': {'dtype': 'float32'}},
                         format="NETCDF4", engine="netcdf4")
    del ds, da, ds_sub
