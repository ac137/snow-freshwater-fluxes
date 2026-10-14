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
##############################################################
# clip_scaled_sit_to_gates.ipynb
# takes scaled sit from cs2 product (scaled by snow products)
# (calculated by scale_sit_by_snow.ipynb)
# and clips to gates, also calculates monthly gate average
# note: scaling has issues currently, but scaling factors are correct

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
# dataset_name = 'nesosim'
# dataset_name = 'legos'
dataset_name = 'amsr2'
# dataset_name = 's3'
# monthly mean
scaled_sit = xr.open_dataset('cs2_sit_scaled/{}_sit_scaled.nc'.format(dataset_name))
# cs2_sit = xr.open_dataset('sit_regridded_cs2/cs2_sit_on_{}_grid.nc'.format(dataset_name))

if dataset_name in ['nesosim','amsr2']:
    # get monthly average
    scaled_sit = scaled_sit.resample(time='MS').mean()
if dataset_name == 'legos':
    scaled_sit = scaled_sit.rename({'xc':'x','yc':'y'})
    x_tmp = scaled_sit['x']
    y_tmp = scaled_sit['y']
    scaled_sit['x']= y_tmp.values
    scaled_sit['y']=x_tmp.values

# %%

# %%
# load shapefiles

shapefile_names_nesosim = ['shapefiles/fram-buffer-50km-3413.shp','shapefiles/davis-buffer-50km-3413.shp']
shapefile_names_smlg = ['shapefiles/fram-buffer-50km-3408.shp','shapefiles/davis-buffer-50km-3408.shp']
shapefile_names_legos = ['shapefiles/fram-buffer-50km-6931.shp','shapefiles/davis-buffer-50km-6931.shp']
shapefile_names_is2 = ['shapefiles/fram-buffer-50km-3411.shp','shapefiles/davis-buffer-50km-3411.shp']

# %%
# need to grab xgrid and ygrid. shouldn't have discarded those
# sit_data_crs = sc.add_crs_nesosim(scaled_sit)
# missing xgrid and ygrid values which are necessary for this. 

# %%
# scaled_sit['sit']

# %%
# scaled_sit

# %%
if dataset_name == 'nesosim':
    # sit_data = xr.open_dataset(
    shapefiles_nesosim = sc.open_shapefiles_from_list(shapefile_names_nesosim)
    # loading a nesosim data file since clipping stripped the x grid coordinates
    snow_data_nesosim = sc.add_crs_and_dates_nesosim(xr.open_dataset('../snow_modelling/nesosim_default_data/NESOSIMv11_01091980-30041981.nc'))
    
    nesosim_sample_data = snow_data_nesosim['snow_depth'][0].copy()
    
    # nesosim_sample_data = sc.add_crs_nesosim(nesosim_sample_data)
    nesosim_sample_data[:] = 1.0
    # does this work without assigning crs or something?
    nesosim_gate_masks = sc.clip_to_shape_list(nesosim_sample_data, shapefiles_nesosim)

    gate_mask_dict = {}
    # todo for refactoring: reduce amount of nesting in dictionaries
    gate_mask_dict['Fram'] = nesosim_gate_masks[0]
    gate_mask_dict['Davis'] = nesosim_gate_masks[1]

elif dataset_name == 'legos':
    shapefiles_legos = sc.open_shapefiles_from_list(shapefile_names_legos)
    legos_sample_data = xr.open_dataset('../other_data/SD_LaKu_LEGOS_NH_Carret_and_Fleury/SD_LaKu_LEGOS_NH_201810.nc')
    legos_sample_data = legos_sample_data['snow_depth_laku_sam'][0].copy()
    legos_sample_data = legos_sample_data.rename({'xc':'x','yc':'y'})
    # legos had indexing issues; need to swap coords
    x_tmp = legos_sample_data['x']
    y_tmp = legos_sample_data['y']
    legos_sample_data['x']= y_tmp.values
    legos_sample_data['y']=x_tmp.values

    legos_sample_data[:]=1.0
    # # need to set spatial dims this way because otherwise it complains
    legos_sample_data = legos_sample_data.rio.set_spatial_dims('y','x')
    legos_sample_data = legos_sample_data.rio.write_crs("epsg:6931",inplace=True)
    legos_gate_masks = sc.clip_to_shape_list(legos_sample_data, shapefiles_legos) 
    gate_mask_dict = {}
    # todo for refactoring: reduce amount of nesting in dictionaries
    gate_mask_dict['Fram'] = legos_gate_masks[0]
    gate_mask_dict['Davis'] = legos_gate_masks[1]

elif dataset_name == 'smlg' or dataset_name == 'amsr2' or dataset_name == 'smlg-m2':
    shapefiles_smlg = sc.open_shapefiles_from_list(shapefile_names_smlg)
    # clip a single day to get a mask that's less computationally expensive to apply
    smlg_sample_data = xr.open_dataset('../SnowModel_data/smlg_small/smlg_1d.nc')
    # write crs
    smlg_sample_data = sc.add_crs_snowmodel(smlg_sample_data)['snod']
    
    smlg_sample_data[:]=1.0
    smlg_gate_masks = sc.clip_to_shape_list(smlg_sample_data, shapefiles_smlg) 

    gate_mask_dict = {}
    # todo for refactoring: reduce amount of nesting in dictionaries
    gate_mask_dict['Fram'] = smlg_gate_masks[0]
    gate_mask_dict['Davis'] = smlg_gate_masks[1]
# elif dataset_name == 'legos_sam':
#     shapefiles_legos = sc.open_shapefiles_from_list(shapefile_names_legos)
#     legos_sample_data = preprocessed_data_dict['snow_depth']['snow_depth'][0].copy()
#     legos_sample_data[:]=1.0
#     # need to set spatial dims this way because otherwise it complains
#     legos_sample_data = legos_sample_data.rio.set_spatial_dims('y','x')
#     legos_sample_data = legos_sample_data.rio.write_crs("epsg:6931",inplace=True)
#     legos_gate_masks = sc.clip_to_shape_list(legos_sample_data, shapefiles_legos) 
#     gate_mask_dict = {}
#     # todo for refactoring: reduce amount of nesting in dictionaries
#     gate_mask_dict['Fram'] = legos_gate_masks[0]
#     gate_mask_dict['Davis'] = legos_gate_masks[1]

elif dataset_name == 's3':
    shapefiles_s3= sc.open_shapefiles_from_list(shapefile_names_legos)
    # clip a single day to get a mask that's less computationally expensive to apply
    s3_sample_data = xr.open_dataset('s3_aggregated/s3_aggregated_snow_depth_2016-10.nc')
    # add coords
    s3_sample_data = sc.add_crs_legos(s3_sample_data)
    s3_sample_data = s3_sample_data['snow_depth'].copy()
    s3_sample_data[:]=1.0
    gate_mask_dict = {}
    gate_mask_dict['Fram'], gate_mask_dict['Davis'] = sc.clip_to_shape_list(s3_sample_data, shapefiles_s3)


# %%
# s3_sample_data

# %%
# gate_clipped_data_0 = scaled_sit.where(nesosim_gate_masks[0].values==1)
gate_clipped_sit = {}
gate_averaged_sit = {}
for k,v in gate_mask_dict.items():
    
    gate_clipped_sit[k] = scaled_sit.where(v.values==1)
    gate_averaged_sit[k] = gate_clipped_sit[k].mean(dim=['x','y']).to_dataframe()


# %%
time_idx = 4

scaled_sit['sit'][time_idx].plot(vmin=0)#[4].plot()
(gate_clipped_sit['Fram']['sit'][time_idx]).plot(cmap='Oranges',vmin=-5,vmax=3) # okay clipping looks accurate
(gate_clipped_sit['Davis']['sit'][time_idx]).plot(cmap='Oranges',vmin=-5,vmax=3) # okay clipping looks accurate

# plt.imshow(nesosim_gate_masks[1],cmap='Reds',origin='lower')
# nesosim_gate_masks[1].plot(cmap='Reds')
plt.show()
scaled_sit['sit'][time_idx].plot(vmin=0)
plt.show()

# %%
gate_average_sit_scaled = pd.concat(gate_averaged_sit,axis=1)#['Fram']['sit'].plot()

# %%
gate_average_sit_scaled['Fram']['sit'].plot()
plt.show()
gate_average_sit_scaled['Davis']['sit'].plot()
plt.show()

# %%
# save output data

gate_average_sit_scaled.to_csv('cs2_sit_scaled/gate_average_monthly_sit_scaled_{}.csv'.format(dataset_name))

# %%
for k,v in gate_clipped_sit.items():
    fn = 'cs2_sit_scaled/{}_sit_scaled_gate_clipped_{}.nc'.format(k, dataset_name)
    sc.save_xr_comp(v, fn)

# %%
print('done')

# %%
