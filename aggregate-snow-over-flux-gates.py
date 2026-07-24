# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
## aggregate-snow-over-flux-gates.ipynb
# notebook to load gate-clipped gridded snow data (generated using batch_clip_to_shapefile_intermediate.ipynb)
# and calculate spatial average over gates (with monthly temporal averaging also if necessary) and uncertainty estimates

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import os
import pyproj
import rioxarray as rio
import geopandas
from shapely.geometry import mapping


import cartopy
import cartopy.crs as ccrs

import shapefile_clipping_helpers as sc

# %%

gate_list = ['Fram','Davis']
variable_list = ['snow_density', 'myi_fraction', 'snow_depth', 'snow_salinity']
# dataset_name = 'nesosim'
# dataset_name = 'smlg'
# dataset_name = 'amsr2'
# dataset_name = 'legos_sam'
dataset_name = 's3'

# is the data monthly; for aggregation purposes
data_is_monthly = {'nesosim':False, 'smlg':False, 'amsr2':False,'legos_sam':True,'s3':True}

# %%
clipped_data_dict = sc.load_clipped_data_to_dict(dataset_name, gate_list, variable_list)


# %%
snow_depth_gate_mean = {}

for g in gate_list:
    snow_depth_gate_mean[g] = clipped_data_dict[g]['snow_depth']['snow_depth'].mean(dim=['x','y']).to_dataframe()['snow_depth']
snow_depth_gate_mean = pd.DataFrame(snow_depth_gate_mean)


# %%
snow_depth_gate_std = {}
for g in gate_list:
    snow_depth_gate_std[g] = clipped_data_dict[g]['snow_depth']['snow_depth'].std(dim=['x','y']).to_dataframe()['snow_depth']
snow_depth_gate_std = pd.DataFrame(snow_depth_gate_std)

# %%
snow_depth_gate_med = {}

for g in gate_list:
    snow_depth_gate_med[g] = clipped_data_dict[g]['snow_depth']['snow_depth'].median(dim=['x','y']).to_dataframe()['snow_depth']
snow_depth_gate_med = pd.DataFrame(snow_depth_gate_med)

# %%
snow_depth_gate_max = {}

for g in gate_list:
    snow_depth_gate_max[g] = clipped_data_dict[g]['snow_depth']['snow_depth'].max(dim=['x','y']).to_dataframe()['snow_depth']
snow_depth_gate_max = pd.DataFrame(snow_depth_gate_max)

# %%
snow_depth_gate_mean.plot() # plot of mean (may be daily or monthly)
plt.show()

# %%
snow_depth_gate_std.plot()
plt.show()

# %%
snow_depth_gate_med.plot() # median
plt.show()

# %%
snow_depth_gate_max.plot() # max
plt.show()

# %%
snow_depth_gate_mean_monthly = snow_depth_gate_mean.resample('MS').mean() # time averaging
snow_depth_gate_mean_monthly.plot()
plt.show()

# %%
snow_depth_gate_std_monthly = snow_depth_gate_std.resample('MS').mean()
snow_depth_gate_std_monthly.plot()
plt.show()

# %%
snow_depth_gate_std_monthly2 = snow_depth_gate_mean.resample('MS').std()
snow_depth_gate_std_monthly2.plot()
plt.show()

# %%
std_errprop = np.sqrt(snow_depth_gate_std_monthly2**2 + snow_depth_gate_std_monthly**2)
std_errprop.plot()
plt.show()

# %%
data_is_monthly['s3']

# %%
if data_is_monthly[dataset_name]:
    # if the data is monthly, save spatial std
    std_to_save = snow_depth_gate_std_monthly
else:
    # if data is daily, also propagate in the time uncertainty
    std_to_save = std_errorprop



# %%
std_to_save.plot()
plt.show()

# %%
snow_depth_gate_med_monthly = snow_depth_gate_med.resample('MS').median()
snow_depth_gate_med_monthly.plot()
plt.show()

# %%
# for g in gate_list:
#     # single-day histograms for each gate
#     clipped_data_dict[g]['snow_depth']['snow_depth'][230].plot.hist()
#     plt.show()

# %%
# calculate snow water equivalent also
swe_gate_mean = {}
rho_w = 1000 # kg/m^3
for g in gate_list:
    swe_gate_mean[g] = (clipped_data_dict[g]['snow_depth']['snow_depth']*clipped_data_dict[g]['snow_density']['snow_density']).mean(dim=['x','y']).to_dataframe(name=g)[g]
swe_gate_mean = pd.DataFrame(swe_gate_mean)

# %%
swe_gate_mean.plot()
plt.show()

# %%
swe_gate_mean_monthly = swe_gate_mean.resample('MS').mean()
swe_gate_mean_monthly.plot()
plt.show()

# %%
snow_depth_gate_mean_monthly.to_hdf('snow_depth_swe_gate_mean_monthly_{}.h5'.format(dataset_name),key='snow_depth')
swe_gate_mean_monthly.to_hdf('snow_depth_swe_gate_mean_monthly_{}.h5'.format(dataset_name),key='swe')


# %%
std_to_save.to_hdf('snow_depth_swe_gate_mean_monthly_{}.h5'.format(dataset_name),key='snow_depth_std')

# %%
