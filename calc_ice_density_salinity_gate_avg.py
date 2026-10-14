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
# load sea ice age



ice_age = xr.open_mfdataset('../other_data/ice_age_nsidc/*.nc/')

# %%
density_myi = 882
density_fyi = 917 #densities from landy et al 2022

# %%
# def sal_fyi():
#     a = 1.0964
#     b = -1.0552
#     c = 4.41272
#     return 1/b + c + (a/b**2)*(np.log(a/(a+b)))
# def sal_myi():
#     a = 0.17083
#     b = 0.92762
#     c = 0.024516
#     return (1/2*a)+(c/(c+1))*((1/b)**(1/c))

# salinity_myi = sal_myi()
# salinity_fyi = sal_fyi()

salinity_myi = 2
salinity_fyi = 5 # both values per https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2005GL023427

# %%
ice_age#['crs'] # sea ice age is epsg 3408 at 12.5 km grid. 

# %%
shapefile_names_3408 = ['shapefiles/fram-buffer-50km-3408.shp','shapefiles/davis-buffer-50km-3408.shp']
gate_shapefiles = sc.open_shapefiles_from_list(shapefile_names_3408)

# %%
ice_age = sc.add_crs_snowmodel(ice_age)

# %%
ice_age

# %%
myi_sample_data = ice_age['age_of_sea_ice'][0].copy()
myi_sample_data[:]=1.0
gate_mask_dict = {}
gate_mask_dict['Fram'], gate_mask_dict['Davis'] = sc.clip_to_shape_list(myi_sample_data, gate_shapefiles)

# %%
gate_mask_dict['Fram'].plot()#[0]
plt.show()
gate_mask_dict['Davis'].plot()#[0]
plt.show()
ice_age['age_of_sea_ice'][0].plot()
plt.show()

# %%
# set where age = 20 as nan

ice_age_values_zero_masked = xr.where(ice_age['age_of_sea_ice']<20, ice_age['age_of_sea_ice'], np.nan)
# (ice_age_values_zero_masked[0]==0).plot() # where values are zero, sea ice concentration is low, need to also set that to nan
ice_age_values_zero_masked = xr.where(ice_age_values_zero_masked>0, ice_age_values_zero_masked, np.nan)

# %%
density_fyi

# %%

# %%
# first just do fyi vs myi I guess

ice_density = xr.where(ice_age['age_of_sea_ice']>1,density_myi, density_fyi)


# %%
ice_salinity = xr.where(ice_age['age_of_sea_ice']>1,salinity_myi, salinity_fyi)

# %%
ice_density[0].plot()

# %%
# now need to mask out
# exclude land
# exclude 

# mask out land
density_values_masked = xr.where(ice_age['age_of_sea_ice']<20, ice_density, np.nan)
ice_age_values_zero_masked = xr.where(ice_age['age_of_sea_ice']<20, ice_age['age_of_sea_ice'], np.nan)
# mask out ice-free regions (i.e. must be greater than zero)
# mask based on ice_age_values_zero_masked since need to exclude the nan assigned to values >= 20 earlier
density_values_masked = xr.where(ice_age_values_zero_masked>0, density_values_masked, np.nan) # this works since np.nan > 0 is false
ice_age_values_zero_masked = xr.where(ice_age_values_zero_masked>0, ice_age_values_zero_masked, np.nan)

# %%
# mask out values for salinity also


# mask out land
salinity_values_masked = xr.where(ice_age['age_of_sea_ice']<20, ice_salinity, np.nan)
ice_age_values_zero_masked = xr.where(ice_age['age_of_sea_ice']<20, ice_age['age_of_sea_ice'], np.nan)
# mask out ice-free regions (i.e. must be greater than zero)
# mask based on ice_age_values_zero_masked since need to exclude the nan assigned to values >= 20 earlier
salinity_values_masked = xr.where(ice_age_values_zero_masked>0, salinity_values_masked, np.nan) # this works since np.nan > 0 is false
ice_age_values_zero_masked = xr.where(ice_age_values_zero_masked>0, ice_age_values_zero_masked, np.nan)

# %%
density_values_masked[0].plot()

# %%
salinity_values_masked[0].plot()
plt.show()

# %%
# 
ice_age_values_zero_masked[0].plot()

# %%
# need to calculate fyi/myi split first, then mask?

# %%
ice_density_fram = xr.where(gate_mask_dict['Fram']>0.5, density_values_masked, np.nan)

# %%
ice_density_davis = xr.where(gate_mask_dict['Davis']>0.5, density_values_masked, np.nan)

# %%
salinity_fram = xr.where(gate_mask_dict['Fram']>0.5, salinity_values_masked, np.nan)
salinity_davis = xr.where(gate_mask_dict['Davis']>0.5, salinity_values_masked, np.nan)

# %%
ice_age_values_zero_masked[0].plot()
ice_density_fram[:,:,0].plot(cmap='Reds')
ice_density_davis[:,:,0].plot(cmap='Reds')
# confirming the gate masking works correctly; yes
# plt.ylim(-2e6,0)
# plt.xlim(-1e6,1e6)
plt.show()


# %%
ice_density_fram[:,:,0].plot()
plt.ylim(-2e6,0)
plt.xlim(-1e6,1e6)
plt.show()

# %%
# for the sea ice age, multiyear ice is defined here as ice_age > 1. ice age = 1 indicates fyi

# %%
density_fram_gate_avg = ice_density_fram.mean(dim=['x','y'])
density_davis_gate_avg = ice_density_davis.mean(dim=['x','y'])


# %%
salinity_fram_gate_avg = salinity_fram.mean(dim=['x','y'])
salinity_davis_gate_avg = salinity_davis.mean(dim=['x','y'])

# %%
density_fram_gate_avg_df = density_fram_gate_avg.to_dataframe()
density_davis_gate_avg_df = density_davis_gate_avg.to_dataframe()

# %%
salinity_fram_gate_avg_df = salinity_fram_gate_avg.to_dataframe()
salinity_davis_gate_avg_df = salinity_davis_gate_avg.to_dataframe()

# %%
salinity_fram_gate_avg_df

# %%
density_fram_gate_avg_df['age_of_sea_ice'].plot() # 
density_davis_gate_avg_df['age_of_sea_ice'].plot()
# plt.ylim(916,917)

# %%
salinity_fram_gate_avg_df['age_of_sea_ice'].plot() # 
salinity_davis_gate_avg_df['age_of_sea_ice'].plot()

# %%
# density_fram_gate_avg_df.drop(columns='crs').rename(columns={'age_of_sea_ice':'sea_ice_density'})

# %%
# density_davis_gate_avg_df.drop(columns='crs').rename(columns={'age_of_sea_ice':'sea_ice_density'})

# %%
si_density_df = pd.concat({'Fram':density_fram_gate_avg_df.drop(columns='crs').rename(columns={'age_of_sea_ice':'sea_ice_density'}),
           'Davis':density_davis_gate_avg_df.drop(columns='crs').rename(columns={'age_of_sea_ice':'sea_ice_density'})},axis=1)

# %%
si_salinity_df = pd.concat({'Fram':salinity_fram_gate_avg_df.drop(columns='crs').rename(columns={'age_of_sea_ice':'sea_ice_salinity'}),
           'Davis':salinity_davis_gate_avg_df.drop(columns='crs').rename(columns={'age_of_sea_ice':'sea_ice_salinity'})},axis=1)

# %%
si_density_df # resample monthly probably for fwf calculation

# %%
si_salinity_df

# %%
# pd.to_datetime(si_density_df.index)

# %%
si_density_df.index = pd.to_datetime(si_density_df.index.values.astype(str))

# %%
si_salinity_df.index = pd.to_datetime(si_salinity_df.index.values.astype(str))

# %%
si_density_df_monthly = si_density_df.resample('MS').mean() 

# %%
si_salinity_df_monthly = si_salinity_df.resample('MS').mean() 

# %%
si_density_df_monthly #

# %%
# fill nan with first-year ice density in case there's some fyi that's missing. assuming any minimal sea ice left over would be fyi. 

si_density_df_monthly = si_density_df_monthly.fillna(density_fyi)

# %%
si_salinity_df_monthly = si_salinity_df_monthly.fillna(salinity_fyi)

# %%
si_density_df_monthly.plot()

# %%
si_salinity_df_monthly.plot()

# %%
si_density_df_monthly.to_csv('monthly_sea_ice_density_from_age.csv')

# %%
si_salinity_df_monthly.to_csv('monthly_sea_ice_salinity_from_age.csv')

# %%
