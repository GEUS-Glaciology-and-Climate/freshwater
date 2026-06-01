#!/usr/bin/env python
# Compute per-cell outlet distance statistics across k values (80, 90, 100).
#
# Reads ./tmp/cells_xy.bsv (produced by k_basin_change.sh) and for each grid
# cell calculates how far the routed outlet moves between k scenarios.
# Writes ./tmp/cells_xy_ice_stats.bsv for re-import into GRASS.

import pandas as pd
import numpy as np

df_xy = pd.read_csv("./tmp/cells_xy.bsv", delimiter="|", usecols=['x', 'y']).astype(int)

header = pd.read_csv("./tmp/cells_xy.bsv", delimiter="|", nrows=1)
cols_in = header.columns[['ice' in _ for _ in header.columns]]
df = pd.read_csv("./tmp/cells_xy.bsv", delimiter="|", usecols=cols_in)

multi_hdr = [np.array([_.split("_")[1] for _ in cols_in]).astype(int),
             [_.split("_")[2] for _ in cols_in]]
df.columns = pd.MultiIndex.from_arrays(multi_hdr, names=['k', 'coord'])

k = np.unique(multi_hdr[0])

for kk in k:
    df[(kk, 'avg_dist')] = (
        ((df.xs('x', level='coord', axis='columns').drop(kk, axis='columns').apply(lambda c: c - df.xs('x', level='coord', axis='columns')[kk]))**2 +
         (df.xs('y', level='coord', axis='columns').drop(kk, axis='columns').apply(lambda c: c - df.xs('y', level='coord', axis='columns')[kk]))**2)**0.5
    ).mean(axis='columns')

    df[(kk, 'max_dist')] = (
        ((df.xs('x', level='coord', axis='columns').drop(kk, axis='columns').apply(lambda c: c - df.xs('x', level='coord', axis='columns')[kk]))**2 +
         (df.xs('y', level='coord', axis='columns').drop(kk, axis='columns').apply(lambda c: c - df.xs('y', level='coord', axis='columns')[kk]))**2)**0.5
    ).max(axis='columns')

df_out = pd.DataFrame()
df_out['avg_avg'] = df.xs('avg_dist', level='coord', axis='columns').mean(axis='columns')
df_out['max_max'] = df.xs('max_dist', level='coord', axis='columns').max(axis='columns')

df_out = df_out.merge(df_xy, left_index=True, right_index=True)
df_out.to_csv("./tmp/cells_xy_ice_stats.bsv", sep="|", float_format='%d')
