#!/usr/bin/env python
# Convert per-basin coverage CSV files (from coverage_calc.sh) to NetCDF.
#
# coverage = covered_cells / (covered_cells + uncovered_cells)
# Output: ./tmp/coverage_<RCM>_<domain>_<k>.nc for each combination.

import pandas as pd
import xarray as xr
import numpy as np

for model in ['RACMO', 'MAR']:
    for domain in ['ice', 'land']:
        for k in [80, 90, 100]:
            if (domain == 'land') & (k != 100):
                continue

            c = 'coverage' + '_' + model + '_' + domain + '_' + str(k)

            df_raw = pd.read_csv('./tmp/' + c + '.csv', index_col=0, names=["c_or_uc", "area"])
            df_raw = df_raw.drop(df_raw[df_raw.index == '*'].index)
            df_raw.index = df_raw.index.astype(int)

            df_c = df_raw[df_raw['c_or_uc'] == '1']\
                .drop(labels='c_or_uc', axis='columns')\
                .rename({'area': 'covered'}, axis='columns')

            df_uc = df_raw[df_raw['c_or_uc'] == '*']\
                .drop(labels='c_or_uc', axis='columns')\
                .rename({'area': 'uncovered'}, axis='columns')

            df = df_c.merge(df_uc, how='outer', left_index=True, right_index=True)\
                     .sort_index()

            df.fillna({'uncovered': 0}, inplace=True)
            df['runoff'] = df['covered'] / (df['covered'] + df['uncovered'])

            cov = df['runoff']
            cov.index.name = 'station'
            da = xr.DataArray.from_series(cov)
            da.to_netcdf('./tmp/' + c + '.nc', mode='w')
