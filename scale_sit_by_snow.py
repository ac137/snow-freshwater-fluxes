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
#####################################################
# scale_sit_by_snow.ipynb
# given regridded sit (regridded from regrid_sit_cs2_to_snow_grids.ipynb)
# outputs sit scaled with assumption of snow loading from different snow products
# saves output in cs2_sit_scaled directory
# files to be next processed in clip_scaled_sit_to_gates.ipynb


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import re

import pyproj
import rioxarray as rio
import geopandas
from shapely.geometry import mapping
import xesmf
import shapefile_clipping_helpers as sc
import fwf_helper_functions as ff

import datetime as dt

import cartopy
import cartopy.crs as ccrs

# %%
# constants

density_water = 1024 # prescribed single value assumed
delta_p = 0.9


# %%
def scale_sit_proportionally(snow_depth_new, snow_depth_cs2, sit, rho_snow, rho_i, rho_w, delta_p):
    '''scale cs2 ice thickness (sit) assuming thickness would be calculated using snow_depth_new
    instead of snow_depth_cs2, all other variables held equal
    use variables from cs2 thickness product: snow_depth_cs2 (snow depth), rho_snow (snow density), 
    rho_i (ice density), rho_w (water density, const), 
    delta_p (mean radar penetration as fraction of snow depth)
    returns scaling term (which is added to ice thickness) and scaled ice thickness
    assumes time is commensurate for snow_depth_new and cs2 terms'''
    snow_depth_diff = snow_depth_new - snow_depth_cs2
    scaling_term = snow_depth_diff*(rho_w - rho_snow - delta_p*rho_w)/(rho_i - rho_w) 
    return scaling_term, scaling_term + sit

def resample_daily_to_cs2_biweekly(sit_time, snow_depth, snow_time_var='day'):
    '''given daily snow dataset, resample to cs2 data at dates sit_time'''
    left_bound_arr = sit_time - pd.Timedelta(6,'D') # subtract 6 days to create bin bounds on left
    # group by bins
    snow_depth_grouped_bins = snow_depth.groupby_bins(snow_time_var, left_bound_arr, right=False).mean()
    snow_depth_grouped_bins = snow_depth_grouped_bins[:-1]# exclude spurious last index (since this was constructed to overshoot)
    # assign appropriate coordinates (since indexed from left, should be able to align first coordinate to left)
    snow_depth_grouped_bins = snow_depth_grouped_bins.assign_coords({'{}_bins'.format(snow_time_var):sit_time[:len(snow_depth_grouped_bins)].values})
    snow_depth_grouped_bins = snow_depth_grouped_bins.rename({'{}_bins'.format(snow_time_var):'time'})
    return snow_depth_grouped_bins

# sit_legos_grid, legos_masked['time']
def resample_to_monthly_time_indices(sit, dates_to_use):
    ''' for resampling sit to work with a monthly snow dataset
    sit: dataset of sea ice thickness, dates_to_use: dataarray of datetimes
    aggregate sit monthly to months with start dates specified by dates_to_use
    for all variables in sit dataset
    returns a dict with variable names as keys'''

    sit_resampled_to_time = {}
    for var_name, values in sit.items():
        print(var_name)
        # get monthly average of sit data
        # from values, get data from start month to end month of dates_to_use (indexing by 'yyyy-mm' strings extracted from time)
        # subsets by dates first using loc to reduce computation time
        data_resampled = values.loc[dates_to_use[0].dt.strftime('%Y-%m').values:dates_to_use[-1].dt.strftime('%Y-%m').values].resample(time='MS').mean()
        # add to dict
        sit_resampled_to_time[var_name] = data_resampled.loc[dates_to_use]
    return sit_resampled_to_time

def create_and_save_ds_sit(sit, scaling_term, snow_dataset_name):
    '''create dataset and save it'''
    ds = xr.Dataset({'sit':sit,'scaling_term':scaling_term})
    sc.save_xr_comp(ds, 'cs2_sit_scaled/{}_sit_scaled.nc'.format(snow_dataset_name))
    return ds



# %%

# %%
############################################ 
# NESOSIM

# could probably refactor this to switch between different datasets here but will leave for now


fn_nesosim = r"../snow_modelling/nesosim_default_data/NESOSIMv11_010920*.nc"

ds_nesosim = xr.open_mfdataset(fn_nesosim)

# add dates to nesosim data
ds_nesosim['day'] = pd.to_datetime(ds_nesosim['day'],format='%Y%m%d')

# load nesosim-grid SIT

sit_nesosim_grid = xr.open_dataset('sit_regridded_cs2/cs2_sit_on_nesosim_grid.nc')

# %%
snow_depth_nesosim = ds_nesosim['snow_depth'].loc['2010-10':'2023-07'] # presumably want two week averages?

# %%
# left_bound_arr = sit_nesosim_grid.time.loc[:'2023-05'] - pd.Timedelta(6,'D') # bounds on the left for aggregation
# nesosim_snow_depth_grouped_bins = snow_depth_nesosim.groupby_bins('day', left_bound_arr, right=False).mean()
# nesosim_snow_depth_grouped_bins = nesosim_snow_depth_grouped_bins[:-1]
# # assign values properly

# nesosim_snow_depth_grouped_bins = nesosim_snow_depth_grouped_bins.assign_coords(day_bins=sit_nesosim_grid.time.loc[:'2023-04'].values)
# nesosim_snow_depth_grouped_bins = nesosim_snow_depth_grouped_bins.rename({'day_bins':'time'})

nesosim_snow_depth_grouped_bins = resample_daily_to_cs2_biweekly(sit_nesosim_grid.time.loc[:'2023-05'], snow_depth_nesosim, snow_time_var='day')

# %%
# now can scale and crop to shapefile etc

scaling_term_nesosim, scaled_sit_nesosim = scale_sit_proportionally(nesosim_snow_depth_grouped_bins, sit_nesosim_grid['snow_depth'], sit_nesosim_grid['sea_ice_thickness'], 
                         sit_nesosim_grid['snow_density'], sit_nesosim_grid['sea_ice_density'], density_water, delta_p)

# %%
# scaled_sit_nesosim
# scaling_term_nesosim

# ds = xr.Dataset({'sit':scaled_sit_nesosim,'scaling_term':scaling_term_nesosim})
# ds

# %%
# save nesosim output
# sc.save_xr_comp(scaling_term_nesosim, 'cs2_sit_scaled/nesosim_scaling_term.nc')
# sc.save_xr_comp(scaled_sit_nesosim, 'cs2_sit_scaled/sit_scaled_by_nesosim.nc')
# print('done')

ds_out = create_and_save_ds_sit(scaled_sit_nesosim, scaling_term_nesosim, 'nesosim')
print('done')

# %%
# scaled_sit_nesosim[7].plot(vmin=0,vmax=4)
# plt.show()
# sit_nesosim_grid['sea_ice_thickness'][7].plot(vmin=0,vmax=4)
# plt.show()

# %%
# scaling_term_nesosim[7].plot()

# %%
# done scaling by nesosim; do this for other products now

# %%
############################################ 
# LEGOS


legos_data = xr.open_mfdataset('../other_data/SD_LaKu_LEGOS_NH_Carret_and_Fleury/*.nc')

# mask legos data to where it's greater/equal than zero
legos_masked = xr.where(legos_data['snow_depth_laku_sam']>=0, legos_data['snow_depth_laku_sam'], np.nan)

sit_legos_grid = xr.open_dataset('sit_regridded_cs2/cs2_sit_on_legos_grid.nc')

 # %%
 # make sure dates are consistent; need to rearrange legos to first date of month

legos_masked['time'] = legos_masked['time'] - pd.Timedelta(14,'D')

# %%
# legos_data

# %%
sit_resampled_legos = resample_to_monthly_time_indices(sit_legos_grid, legos_masked['time']) # resample sit to monthly indices

# %%
# sit_resampled_legos.keys()

# %%
# sit_resampled_legos['snow_depth']

# %%
# legos_masked

# %%
# scale sit by legos

scaling_term_legos, sit_scaled_by_legos = scale_sit_proportionally(legos_masked.values, sit_resampled_legos['snow_depth'], sit_resampled_legos['sea_ice_thickness'], 
                                               sit_resampled_legos['snow_density'], sit_resampled_legos['sea_ice_density'],  density_water, delta_p)


# %%
# save output
ds_out = create_and_save_ds_sit(sit_scaled_by_legos, scaling_term_legos, 'legos')
print('done')

# %%
# sit_scaled_by_legos[4].plot(vmin=0,vmax=5)

# %%
# sit_resampled_legos['sea_ice_thickness'][4].plot(vmin=0,vmax=5)

# %%
# (sit_scaled_by_legos[4] - sit_resampled_legos['sea_ice_thickness'][4]).plot()
# plt.show()

# %%
# once sit is resampled, just get onto a common time index and then scale from there
# seems to take a long time to resample

# %%
############################################ 
# S3/NaRRS

s3_data_snow = xr.open_mfdataset('s3_aggregated/s3_aggregated_snow_depth_20*.nc')
sit_s3 = xr.open_dataset('sit_regridded_cs2/cs2_sit_on_s3_grid.nc')

# %%
# s3_data_snow

# %%
sit_resampled_s3 = resample_to_monthly_time_indices(sit_s3, s3_data_snow['time'])

# %%
scaling_term_s3, sit_scaled_by_s3 = scale_sit_proportionally(s3_data_snow['snow_depth'], sit_resampled_s3['snow_depth'], sit_resampled_s3['sea_ice_thickness'], 
                                               sit_resampled_s3['snow_density'], sit_resampled_s3['sea_ice_density'],  density_water, delta_p)


# %%
ds_out = create_and_save_ds_sit(sit_scaled_by_s3, scaling_term_s3, 's3')
print('done')

# %%
# sit_scaled_by_s3[4].plot()
# plt.show()

# sit_resampled_s3['sea_ice_thickness'][4].plot()
# plt.show()

# %%
# (sit_resampled_s3['sea_ice_thickness'][4] - sit_scaled_by_s3[4]).plot()
# plt.gca().set_facecolor('#cccccc')
# plt.show()


# %%
################################################
# AMSR2/5VLSTM

sit_amsr_grid = xr.open_dataset('sit_regridded_cs2/cs2_sit_on_smlg_grid.nc')
amsr_da = xr.open_dataset('../snow_modelling/model_snow_obs_comparisons/pmw_amsr2_li_smlg_grid.nc')

# %%
# amsr_da#['date'] # daily needs to be resampled to biweekly as was done with nesosim, I guess

# %%




# %%
# left_bound_arr = sit_amsr_grid.time.loc[:'2020-04'] - pd.Timedelta(6,'D') # bounds on the left for aggregation
# left_bound_arr
# nesosim_snow_depth_grouped_bins = snow_depth_nesosim.groupby_bins('day', left_bound_arr, right=False).mean()
# nesosim_snow_depth_grouped_bins = nesosim_snow_depth_grouped_bins[:-1]
# # assign values properly

# nesosim_snow_depth_grouped_bins = nesosim_snow_depth_grouped_bins.assign_coords(day_bins=sit_nesosim_grid.time.loc[:'2023-04'].values)
# nesosim_snow_depth_grouped_bins = nesosim_snow_depth_grouped_bins.rename({'day_bins':'time'})

# %%
# amsr_snow_depth_biweekly # see if this resampling actually works?

# %%
sit_time = sit_amsr_grid.time.loc['2012-10':'2020-05']
snow_depth = amsr_da['snow_depth'][:,0,:,:] # get the band index there
snow_time_var='date'

# left_bound_arr = sit_time - pd.Timedelta(6,'D') # subtract 6 days to create bin bounds on left


# snow_depth_grouped_bins = snow_depth.groupby_bins(snow_time_var, left_bound_arr, right=False).mean()
# snow_depth_grouped_bins = snow_depth_grouped_bins[:-1]# exclude spurious last index (since this was constructed to overshoot)
# # assign appropriate coordinates (since indexed from left, should be able to align first coordinate to left)
# snow_depth_grouped_bins = snow_depth_grouped_bins.assign_coords({'{}_bins'.format(snow_time_var):sit_time[:len(snow_depth_grouped_bins)].values})
# snow_depth_grouped_bins = snow_depth_grouped_bins.rename({'{}_bins'.format(snow_time_var):'time'})


amsr2_snow_depth_resampled = resample_daily_to_cs2_biweekly(sit_time, snow_depth, snow_time_var='date')/100 # scale by 100 to convert units

# %%
# snow_depth_grouped_bins.loc['2013-01-07'].plot()
# plt.show()


# %%
# snow_depth_grouped_bins.time

# %%
# sit_amsr_grid.sel(time=snow_depth_grouped_bins.time)##.loc['2012-10':'2020-04']

# %%
time_val = amsr2_snow_depth_resampled.time

scaling_term_amsr2, sit_scaled_by_amsr2 = scale_sit_proportionally(amsr2_snow_depth_resampled, sit_amsr_grid.sel(time=time_val)['snow_depth'], 
                                                             sit_amsr_grid.sel(time=time_val)['sea_ice_thickness'], 
                                               sit_amsr_grid.sel(time=time_val)['snow_density'], 
                                                                   sit_amsr_grid.sel(time=time_val)['sea_ice_density'],  density_water, delta_p)


# %%
ds_out = create_and_save_ds_sit(sit_scaled_by_amsr2, scaling_term_amsr2, 'amsr2')
print('done')

# %%
# sit_scaled_by_amsr2.loc['2012-12-07'].plot()
# plt.show()
# sit_amsr_grid['sea_ice_thickness'].loc['2012-12-07'].plot()
# plt.show()

# %%
# (sit_amsr_grid['sea_ice_thickness'].loc['2012-12-07']-sit_scaled_by_amsr2.loc['2012-12-07']).plot()
# plt.show()


# %%
