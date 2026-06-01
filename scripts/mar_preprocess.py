#!/usr/bin/env python
# Extract MAR ice and land runoff to per-year NetCDF files in ./tmp/.
#
# Reads MARv3.14 daily files from $DATADIR, applies the 2D area correction
# (from area_error.sh), masks to ice or land pixels, and writes one NetCDF
# per year per domain. Already-existing output files are skipped.

import xarray as xr
import numpy as np
import os
import glob
from dask.diagnostics import ProgressBar

DATADIR = os.environ['DATADIR']
ROOT = DATADIR + "/MAR/3.14-daily-1km"
area = xr.open_dataset("./tmp/err_2D_area_MAR.nc")

filelist = np.sort(glob.glob(ROOT + "/MARv3.14-daily-ERA5-????.nc"))
years = [_.split("-")[-1].split(".")[0] for _ in filelist]

# Ice runoff (variable RUcorr, mask MSK >= 2)
for y in years:
    outfile = "./tmp/MAR_runoff_ice_" + y + ".nc"
    if os.path.exists(outfile):
        continue

    infile = ROOT + "/MARv3.14-daily-ERA5-" + y + ".nc"
    print("infile:", infile, "outfile:", outfile)
    ds = xr.open_mfdataset(infile, combine='by_coords', engine='h5netcdf', parallel=False)
    ds['RUcorr'] = ds['RUcorr'].where((ds['RUcorr'] >= ds['RUcorr'].attrs['actual_range'][0]) &
                                      (ds['RUcorr'] <= ds['RUcorr'].attrs['actual_range'][1]))
    da = (ds['RUcorr'] * (ds['MSK'] >= 2) * area['Band1'].values)
    da.variable.attrs = {'units': 'mmWE/day', 'long_name': 'Water runoff', 'standard_name': 'Water runoff'}
    ds_sub = da.to_dataset(name='runoff')
    delayed_obj = ds_sub.to_netcdf(outfile, compute=False)
    with ProgressBar():
        results = delayed_obj.compute()
    del ds, da, ds_sub, delayed_obj

# Land runoff (variable RU2, mask MSK == 1)
for y in years:
    outfile = "./tmp/MAR_runoff_land_" + y + ".nc"
    if os.path.exists(outfile):
        continue

    infile = ROOT + "/MARv3.14-daily-ERA5-" + y + ".nc"
    print("infile:", infile, "outfile:", outfile)
    ds = xr.open_mfdataset(infile, combine='by_coords', engine='h5netcdf', parallel=False)
    ds['RU2'] = ds['RU2'].where((ds['RU2'] >= ds['RU2'].attrs['actual_range'][0]) &
                                (ds['RU2'] <= ds['RU2'].attrs['actual_range'][1]))
    da = ds['RU2'] * (ds['MSK'] == 1) * area['Band1'].values
    da.variable.attrs = {'units': 'mmWE/day', 'long_name': 'Water runoff', 'standard_name': 'Water runoff'}
    ds_sub = da.to_dataset(name='runoff')
    delayed_obj = ds_sub.to_netcdf(outfile, compute=False)
    with ProgressBar():
        results = delayed_obj.compute()
    del ds, da, ds_sub, delayed_obj
