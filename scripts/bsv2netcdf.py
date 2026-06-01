#!/usr/bin/env python
# Convert one BSV discharge file to a CF/ACDD-compliant NetCDF file.
#
# Usage: python scripts/bsv2netcdf.py dat/MAR_runoff_ice_2000_100.bsv
#
# Output goes to freshwater/<domain>_<k>/discharge/<RCM>_<year>.nc

import numpy as np
import pandas as pd
import xarray as xr
import datetime
import subprocess
import sys
import os
import uuid

f = sys.argv[1]

RCM, var, domain, year, k = os.path.basename(os.path.splitext(f)[0]).split("_")

OUTDIR = "./freshwater/" + domain + "_" + k + "/discharge"
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR, exist_ok=True)
OUTBASE = RCM + "_" + year
OUTFILE = OUTDIR + "/" + OUTBASE + ".nc"

BSVFILE = f

df = pd.read_csv(BSVFILE, index_col=0, delimiter="|")
df.index = pd.to_datetime(df.index)

RCM, var, domain, year, k = os.path.basename(os.path.splitext(BSVFILE)[0]).split("_")

metadir = domain + "_" + k
meta = pd.read_csv("freshwater/" + metadir + "/outlets.csv", index_col=0).T
meta = meta[df.columns.astype(int)]

ds = xr.Dataset()

ds["station"] = (("station"), df.columns.astype(str).astype(np.int32))
ds["time"] = (("time"), df.index)

ds["station"].attrs["long_name"] = "outlet id"
ds["station"].attrs["cf_role"] = "timeseries_id"

ds["time"].attrs["long_name"] = "time of measurement"
ds["time"].attrs["standard_name"] = "time"
ds["time"].attrs["axis"] = "T"

# gridsize = 100 x 100 m; convert mm/day to m3/s
CONV = 1E-3 * 100 * 100 / 86400
ds["discharge"] = (("station", "time"), df.values.T * CONV)
ds["discharge"].attrs["long_name"] = RCM + " discharge"
ds["discharge"].attrs["standard_name"] = "water_volume_transport_in_river_channel"
ds["discharge"].attrs["units"] = "m3 s-1"
ds["discharge"].attrs["coordinates"] = "lat lon alt station"
ds["discharge"].attrs['coverage_content_type'] = 'modelResult'

ds["lat"] = (("station"), meta.loc['lat'].astype(np.float32))
ds["lat"].attrs.update({"long_name": "latitude", "standard_name": "latitude",
                         "units": "degrees_north", "axis": "Y"})

ds["lon"] = (("station"), meta.loc['lon'].astype(np.float32))
ds["lon"].attrs.update({"long_name": "longitude", "standard_name": "longitude",
                         "units": "degrees_east", "axis": "X"})

ds["alt"] = (("station"), meta.loc['elev'].astype(np.int32))
ds["alt"].attrs.update({"long_name": "height_above_mean_sea_level", "standard_name": "altitude",
                         "units": "m", "positive": "up", "axis": "Z"})

ds["Z2012_sector"] = (("station"), meta.loc["Z2012_sector"].replace(np.nan, value=-1).astype(np.int32))
ds["Z2012_sector"].attrs["long_name"] = "Zwally (2012) sector nearest outlet"
ds["Z2012_sector_dist"] = (("station"), meta.loc["Z2012_sector_dist"].replace(np.nan, value=-1).astype(np.int32))
ds["Z2012_sector_dist"].attrs["long_name"] = "Zwally (2012) sector nearest outlet distance"

ds["M2019_ID"] = (("station"), meta.loc["M2019_ID"].replace(np.nan, value=-1).astype(np.int32))
ds["M2019_ID"].attrs["long_name"] = "Mouginot (2019) basin nearest outlet ID"
ds["M2019_ID_dist"] = (("station"), meta.loc["M2019_ID_dist"].replace(np.nan, value=-1).astype(np.int32))
ds["M2019_ID_dist"].attrs["long_name"] = "Mouginot (2019) basin nearest outlet ID distance"
ds["M2019_basin"] = (("station"), meta.loc["M2019_basin"])
ds["M2019_basin"].attrs["long_name"] = "Mouginot (2019) basin nearest outlet name"
ds["M2019_region"] = (("station"), meta.loc["M2019_region"])
ds["M2019_region"].attrs["long_name"] = "Mouginot (2019) region nearest outlet"

ds["M2020_gate"] = (("station"), meta.loc["M2020_gate"].replace(np.nan, value=-1).astype(np.int32))
ds["M2020_gate"].attrs["long_name"] = "Mankoff (2020) gate nearest outlet"
ds["M2020_gate_dist"] = (("station"), meta.loc["M2020_gate_dist"].replace(np.nan, value=-1).astype(np.int32))
ds["M2020_gate_dist"].attrs["long_name"] = "Mankoff (2020) gate nearest outlet distance"

ds["B2015_name"] = (("station"), meta.loc["B2015_name"])
ds["B2015_name"].attrs["long_name"] = "Bjørk (2015) Official Name gate nearest outlet"
ds["B2015_dist"] = (("station"), meta.loc["B2015_dist"].astype(np.int32))
ds["B2015_dist"].attrs["long_name"] = "Bjørk (2015) Official Name point distance"

for v in ["B2015_dist", "M2019_ID", "M2019_ID_dist", "M2020_gate",
          "M2020_gate_dist", "Z2012_sector", "Z2012_sector_dist"]:
    ds[v].attrs["coverage_content_type"] = "auxiliaryInformation"
    ds[v].attrs["units"] = "N/A"

if "ice" in BSVFILE:
    ds["coast_id"] = (("station"), np.int32(meta.loc["coast_id"].astype(np.float32).replace(np.nan, value=-1)))
    ds["coast_id"].attrs.update({"long_name": "ID of coastal outlet", "units": "N/A",
                                  "coverage_content_type": "auxiliaryInformation"})

    ds["coast_lat"] = (("station"), meta.loc["coast_lat"].astype(np.float32))
    ds["coast_lat"].attrs.update({"long_name": "latitude", "standard_name": "latitude",
                                   "units": "degrees_north", "axis": "Y"})

    ds["coast_lon"] = (("station"), meta.loc["coast_lon"].astype(np.float32))
    ds["coast_lon"].attrs.update({"long_name": "longitude", "standard_name": "longitude",
                                   "units": "degrees_east", "axis": "X"})

    ds["coast_alt"] = (("station"), meta.loc["elev"].astype(np.float32))
    ds["coast_alt"].attrs.update({"long_name": "height_above_mean_sea_level", "standard_name": "altitude",
                                   "units": "m", "positive": "up", "axis": "Z"})

kw = ["GCMDSK:EARTH SCIENCE > CRYOSPHERE > GLACIERS/ICE SHEETS > ICE SHEETS > ICE SHEET MEASUREMENTS",
      "GCMDSK:EARTH SCIENCE > CRYOSPHERE > GLACIERS/ICE SHEETS > GLACIER MASS BALANCE/ICE SHEET MASS BALANCE",
      "GCMDSK:EARTH SCIENCE > CLIMATE INDICATORS > ATMOSPHERIC/OCEAN INDICATORS > FRESH WATER RIVER DISCHARGE",
      "GCMDSK:EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > SURFACE WATER > SURFACE WATER PROCESSES/MEASUREMENTS > DISCHARGE/FLOW",
      "GCMDSK:EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > SURFACE WATER > SURFACE WATER PROCESSES/MEASUREMENTS > DISCHARGE/FLOW > FLOW VELOCITY"]
ds.attrs["keywords"] = ", ".join(kw)
ds.attrs["keywords_vocabulary"] = "GCMDSK:GCMD Science Keywords:https://gcmd.earthdata.nasa.gov/kms/concepts/concept_scheme/sciencekeywords"

ds.attrs["id"] = "dk.geus.promice:" + str(uuid.uuid3(uuid.NAMESPACE_DNS, BSVFILE))
ds.attrs["naming_authority"] = "dk.geus.promice"
ds.attrs["history"] = "Generated on " + datetime.datetime.utcnow().isoformat()
ds.attrs["processing_level"] = "Level 3"
ds.attrs["acknowledgement"] = "The Programme for Monitoring of the Greenland Ice Sheet (PROMICE)"
ds.attrs["license"] = "Creative Commons Attribution 4.0 International (CC-BY-4.0) https://creativecommons.org/licenses/by/4.0"
ds.attrs["standard_name_vocabulary"] = "CF Standard Name Table (v77, 19 January 2021)"
ds.attrs["date_created"] = str(datetime.datetime.now().isoformat())
ds.attrs["creator_name"] = "Ken Mankoff"
ds.attrs["creator_email"] = "kdm@geus.dk"
ds.attrs["creator_url"] = "http://promice.org"
ds.attrs["institution"] = "GEUS"
ds.attrs["publisher_name"] = "GEUS"
ds.attrs["publisher_email"] = "info@promice.dk"
ds.attrs["publisher_url"] = "http://promice.dk"

ds.attrs["geospatial_bounds"] = "POLYGON((" + \
    f'{ds["lat"].min().values} {ds["lon"].min().values}, ' + \
    f'{ds["lat"].min().values} {ds["lon"].max().values}, ' + \
    f'{ds["lat"].max().values} {ds["lon"].max().values}, ' + \
    f'{ds["lat"].max().values} {ds["lon"].min().values}, ' + \
    f'{ds["lat"].min().values} {ds["lon"].min().values}))'
ds.attrs["geospatial_bounds_crs"] = "EPSG:4326"
ds.attrs["geospatial_lat_min"] = ds["lat"].min().values
ds.attrs["geospatial_lat_max"] = ds["lat"].max().values
ds.attrs["geospatial_lon_min"] = ds["lon"].min().values
ds.attrs["geospatial_lon_max"] = ds["lon"].max().values
ds.attrs["geospatial_vertical_min"] = ds["alt"].min().values
ds.attrs["geospatial_vertical_max"] = ds["alt"].max().values
ds.attrs["geospatial_vertical_positive"] = "up"
ds.attrs["time_coverage_start"] = str(ds["time"][0].values)
ds.attrs["time_coverage_end"] = str(ds["time"][-1].values)
ds.attrs["time_coverage_duration"] = pd.Timedelta((ds["time"][-1] - ds["time"][0]).values).isoformat()
ds.attrs["time_coverage_resolution"] = pd.Timedelta((ds["time"][1] - ds["time"][0]).values).isoformat()

ds.attrs["creator_type"] = "person"
ds.attrs["creator_institution"] = "GEUS"
ds.attrs["publisher_type"] = "institution"
ds.attrs["publisher_institution"] = "GEUS"
ds.attrs["program"] = "PROMICE"
ds.attrs["contributor_name"] = ""
ds.attrs["contributor_role"] = ""
ds.attrs["geospatial_lat_units"] = "degrees_north"
ds.attrs["geospatial_lon_units"] = "degrees_east"
ds.attrs["references"] = "Mankoff, Kenneth D., Noël, Brice, Fettweis, Xavier, Ahlstrøm, Andreas P., Colgan, William, Kondo, Ken, Langley, Kirsty, Sugiyama, Shin, van As, Dirk, Fausto, Robert S.: Greenland liquid water discharge from 1958 through 2019 , Earth System Science Data 12(4), 2811–2841, https://doi.org/10.5194/essd-12-2811-2020, 2020"

bib = """@article{mankoff_2020_liquid,
author    = {Mankoff, Kenneth D. and Noël, Brice and Fettweis, Xavier and Ahlstrøm, Andreas P. and
Colgan, William and Kondo, Ken and Langley, Kirsty and Sugiyama, Shin and van As,
Dirk and Fausto, Robert S.},
title     = {{G}reenland liquid water discharge from 1958 through 2019},
journal   = {Earth System Science Data},
year      = 2020,
volume    = 12,
number    = 4,
pages     = {2811–2841},
month     = 11,
ISSN      = {1866-3516},
DOI       = {10.5194/essd-12-2811-2020},
publisher = {Copernicus GmbH}}"""
ds.attrs["references_bib"] = bib
ds.attrs["project"] = "PROMICE"

ds.attrs["featureType"] = "timeSeries"
ds.attrs["title"] = "Greenland discharge"
ds.attrs["comment"] = "Greenland discharge"
ds.attrs["summary"] = "Greenland RCM discharge at basin outlets"
ds.attrs["keywords_other"] = "Hydrology; Greenland; Runoff; Discharge; Freshwater"
ds.attrs["Conventions"] = "CF-1.8, ACDD-1.3"
ds.attrs["source"] = "git commit: " + \
    subprocess.check_output(["git", "describe", "--always"]).strip().decode("UTF-8")
ds.attrs["DOI"] = "10.22008/promice/freshwater"
ds.attrs["product_version"] = "daily updates"

# apply coverage scaling
cf = "coverage_" + RCM + "_" + domain + "_" + k + ".nc"
c = xr.open_dataset("./tmp/" + cf)
c = c.where(c['runoff'] != 0, np.nan, drop=False)
ds['discharge'].values = ds['discharge'] / c['runoff'].expand_dims(time=ds["time"])

enc = {}
for v in ['discharge', 'alt', 'Z2012_sector', 'M2019_ID', 'M2020_gate']:
    enc[v] = {'zlib': True, 'complevel': 2}
enc['time'] = {'dtype': 'i4'}
enc['station'] = {'dtype': 'i4'}

ds.to_netcdf(OUTFILE, unlimited_dims='time', encoding=enc, engine='netcdf4', mode='w')
