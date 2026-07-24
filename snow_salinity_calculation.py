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
## snow_salinity_calculation.ipynb
# script for estimating snow bulk salinity by prescribing by layer for all layers below 16 cm
# (snow above 16 cm assumed to have zero thickness)
# salinity by layer estimated based on observed values over FYI from figure S2 in Nandan et al., 2017 (https://doi.org/10.1002/2017GL074506)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import os

import shapefile_clipping_helpers as sc

# %%
nesosim_da = xr.open_mfdataset('../nesosim_default_data/*.nc')


# %%
smlg_da = xr.open_mfdataset('../SM_snod_ERA5_01Aug1980-31Jul2021_v01.nc')

# %%
smlg_da = xr.open_mfdataset('../SM_snod_MERRA2_01Aug1980-31Jul2021_v01.nc') # for MERRA2

# %%
amsr_da = xr.open_dataset('../pmw_amsr2_li_smlg_grid.nc')

# %%
legos_da = xr.open_mfdataset('../SD_LaKu_LEGOS_NH_Carret_and_Fleury/*.nc')

# %%
s3_da = xr.open_mfdataset('./s3_aggregated/s3_aggregated_snow_depth_20*.nc')
# none of these are less than zero

# %%
# icesat 2 dataset; using to get MW99 snow depth and density - not used in publication
# mw_da = xr.open_mfdataset('../IS2SITMOGR4_v4/IS2SITMOGR4_01_*.nc')
# mw_da = mw_da.sortby('time')

# %%
## load salinity by height (based on Nandan et al., 2017)
salinity_df = pd.read_csv('snow_salinity_by_height_for_processing.csv')

# %%
# this gives a list of salinities by height in cm, use in get_salinity_by_height
salinity_list = salinity_df['salinity (ppt)']

# this function works for a single value but not for all values in an xarray
def get_salinity_by_height(height):#, salinity_list):
    ''' given height (float) in m, return bulk salinity for snowpack in ppt;
    assumes existence of salinity_list'''
    height *= 100 #convert to cm
    max_salinity_height = len(salinity_list) # in cm

    # if height exceeds 16 cm (max height), calculate bulk salinity for lower
    # layers and do weighted average with 
    if height > max_salinity_height:
        height_nonsaline = height - max_salinity_height
        salinity_lower_layers = np.sum(salinity_list)/len(salinity_list)
        total_salinity = salinity_lower_layers*max_salinity_height/height
        return total_salinity # leaving return statement here for slightly faster execution (avoid hitting the else)
    elif height > 0:
        height_int = int(np.round(height)) # can be up to 16
        if height_int==0:
            # if it rounds down to zero, set to lowest 1 cm
            height_int = 1
        salinity_list_subset = salinity_list[:height_int]
        

        total_salinity = np.sum(salinity_list_subset)/len(salinity_list_subset)
        return total_salinity
        
    else:
        return 0. # no salinity

def calculate_salinity_by_layer_height(n, salinity_list):
    '''given n (int, 0<n<=len(salinity_list)), salinity_list
    (list of float), return calculated bulk salinity for snow 
    with thickness n cm, assuming salinity_list contains salinity
    for first n layers of snowpack (layer thickness=1 cm)'''
    salinity_list_subset = salinity_list[:int(n)]
    return np.sum(salinity_list_subset)/len(salinity_list_subset)
    # n integer between 1 and 16
    # does this introduce an off-by-one error if height = 16? double-check
    # actually I think it's fine because then it's indexed as [:16] which should be
    # the whole list

def get_salinity_by_height_xr(height, salinity_list):
    ''' given height, an xarray containing snow thickness in m, 
    salinity_list; a list of salinity values by layer (from obs,
    lowest to highest, assuming 1 cm thick layers)
    return xarray of bulk salinity for the snowpack in ppt calculated
    from observations'''
    height_xr = height*100 #convert to cm

    # blank xarray for storing salinity
    # doing this so it has the same shape as the snow thickness xarray
    salinity_xr_blank = height*0

    # height above which salinity is set to zero
    max_salinity_height = len(salinity_list) # in cm, should be 16

    # salinity for lower layers; use this in weighted average for
    # layers with height > 16 cm
    lower_layer_bulk_salinity = np.sum(salinity_list)/len(salinity_list)

    # need to calculate separately for grid cells above max salinity height 
    # and below max salinity height

    # doing this in somewhat a strange order because xarray was complaining otherwise

    ######### SALINITY ABOVE MAX HEIGHT (salinity = 0)
    
    # condition for where to populate salinity_xr with these values
    heights_above_max_cond = height_xr > max_salinity_height
    # wait we need to account for non-saline values when above threshold height

    # see which values in height_xr satisfy this condition
    # heights_above_max has height if height > max_salinity_height, otherwise has nan
    heights_above_max = xr.where(heights_above_max_cond, height_xr, np.nan)
    # calculate total salinity for values above max,
    # later will assign these values to salinity_xr
    total_salinity_above_max = lower_layer_bulk_salinity*max_salinity_height/heights_above_max
    # populate the salinity xarray after

    # specify the salinity xarray 
    salinity_xr = salinity_xr_blank

    ############# SALINITY BELOW MAX HEIGHT

    # find which heights satisfy this condition
    height_below_max_cond = height_xr <= max_salinity_height
    
    height_below_max = xr.where(height_below_max_cond, height_xr, 0) # zero denotes other heights here

    # round height to nearest integer to be able to select layers
    height_below_max_rounded = np.round(height_below_max).astype(int)

    # go through snow thickness from 1 to 16, calculate salinity for that height,
    # find which layers have that height, and then assign that salinity to 
    # those layers

    for i in range(1,17):
        # calculate salinity for specific height

        salinity_for_height = calculate_salinity_by_layer_height(i, salinity_list)

        # find all indices that have height i and update salinity_xr to assign corresponding salinity there
        matching_array_cond = height_below_max_rounded==i
        salinity_xr = xr.where(matching_array_cond,salinity_for_height,salinity_xr)

    # for all salinity values with height_below_max_rounded=0, assign salinity as if equal to 1 cm
    # (so snow with thickness between 0 and 1 cm has salinity of 1-cm thick snow)
    # note: this also assigns that value for salinities above max also but those will be overwritten at next step

    salinity_xr = xr.where(height_below_max_rounded==0, calculate_salinity_by_layer_height(1, salinity_list),salinity_xr)

    # for values with height above max, use the calculated salinity now
    salinity_xr = xr.where(heights_above_max_cond,total_salinity_above_max,salinity_xr)

    # now wherever there are actual zero height values, set salinities to zero
    salinity_xr = xr.where(height_xr>0, salinity_xr, 0)

    return salinity_xr



# %%
depth_date_range = nesosim_da['snow_depth'].loc['20161001':]
salinity_date_range = get_salinity_by_height_xr(depth_date_range, salinity_list).to_dataset(name='snow_salinity')
sc.save_xr_comp(salinity_date_range, 'nesosim_snow_salinity_estimate.nc')

# %%
depth_smlg_date_range = smlg_da['snod'].loc['2016-01-01':]
salinity_date_range_smlg = get_salinity_by_height_xr(depth_smlg_date_range, salinity_list).to_dataset(name='snow_salinity')

# sc.save_xr_comp(salinity_date_range_smlg, 'smlg_snow_salinity_estimate.nc')
# FOR MERRA2 SMLG
sc.save_xr_comp(salinity_date_range_smlg, 'smlg_snow_salinity_estimate_m2.nc')

# %%
depth_amsr2_date_range = amsr_da['snow_depth'].loc['2016-01':]/100. # convert to m
salinity_date_range_amsr2 = get_salinity_by_height_xr(depth_amsr2_date_range, salinity_list).to_dataset(name='snow_salinity')

sc.save_xr_comp(salinity_date_range_amsr2, 'amsr2_li_snow_salinity_estimate.nc')


# %%
depth_legos_sam_date_range = legos_df['snow_depth_laku_sam']
salinity_date_range_legos_sam = get_salinity_by_height_xr(depth_legos_sam_date_range, salinity_list).to_dataset(name='snow_salinity')

sc.save_xr_comp(salinity_date_range_legos_sam, 'legos_sam_snow_salinity_estimate.nc')


# %%
depth_s3_date_range = s3_da['snow_depth'] # I think the date range is fine here
salinity_date_range_s3 = get_salinity_by_height_xr(depth_s3_date_range, salinity_list).to_dataset(name='snow_salinity')

sc.save_xr_comp(salinity_date_range_s3, 's3_snow_salinity_estimate.nc')

# %%
# depth_mw99_date_range = mw_da['snow_depth_mw99_int']
# salinity_date_range_mw99 = get_salinity_by_height_xr(depth_mw99_date_range, salinity_list).to_dataset(name='snow_salinity')
# sc.save_xr_comp(salinity_date_range_mw99, 'mw99_snow_salinity_estimate.nc')

# %%
print('saved')
