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
# load data and regrid

# %%
# need file path first
# sit_dir_path = '../other_data/landy_sit_v2-0/'
sit_dir_path = '../other_data/landy_sit_v2/year_round_sit_v2p0/'

sit_file_year = 2019

# sit_filename = 'developers_ubristol_cryosat2_seaicethickness_{}_nh_80km_v1p7.nc'.format(sit_file_year)
# sit_filename = 'ubristol_cryosat2_seaicethickness_nh_80km_v1p7.nc'
sit_filename = 'uit_sit_smlg_biweekly_v2north_polarstereo_80km_20190107_v2p0.nc'
# sit_file_path = os.path.join(sit_dir_path, sit_filename)
sit_file_path = os.path.join(sit_dir_path, '*.nc')

# %%
sit_data = xr.open_mfdataset(sit_file_path)#,decode_times=False) # v2 has decoded times

# %%
sit_data['longitude'][0]

# %%
# def adjust_time(sit_data,time_varname='Time'):
#     '''adjust time, modifies in place'''
#     date_initial = np.datetime64('-001-12-31')
#     times_list = []
#     for t in sit_data[time_varname].values:
#         delta_t = np.timedelta64(int(t), 'D')
#         times_list.append(date_initial + delta_t)
#     # sit_data[time_varname] = times_list
#     times_list = np.array(times_list)
#     sit_data[time_varname].values = times_list


# %%
# load nesosim data for regridding

# nesosim and smlg files
root_path = r"../snow_modelling/nesosim_default_data/"
year_file = "NESOSIMv11_01092018-30042019.nc"

sm_data_fn = "../SnowModel_data/SM_snod_ERA5_01Aug1980-31Jul2021_v01.nc"

# %%
fn_nesosim = os.path.join(root_path,year_file)

ds_nesosim = xr.open_dataset(fn_nesosim)

# date adjustment probably not necessary but leaving in either way
ds_nesosim['day'] = pd.to_datetime(ds_nesosim['day'],format='%Y%m%d')

# %%
# LEGOS
legos_da = xr.open_dataset('../other_data/SD_LaKu_LEGOS_NH_Carret_and_Fleury/SD_LaKu_LEGOS_NH_201901.nc')

# %%
legos_da

# %%
# sentinel 3 
s3_data = xr.open_dataset('s3_aggregated/s3_aggregated_snow_depth_{}.nc'.format('2018-10'))

# %%
# generate grid lat/lon for smlg
snowmodel_data = xr.open_dataset(sm_data_fn)
lons_sm, lats_sm, proj_sm, sm_grid = sc.generate_snowmodel_grid(snowmodel_data)

# %%
# set up regridding next

grid_nesosim = {"lon": ds_nesosim['longitude'].values, "lat": ds_nesosim['latitude'].values}
# grid_ice_thickness = {"lon":sit_data['Longitude'].values, "lat":sit_data['Latitude'].values}
grid_ice_thickness = {"lon":sit_data['longitude'][0].values, "lat":sit_data['latitude'][0].values} # zero indices to be used only when using mfdataset
grid_legos = {'lon':legos_da['longitude'],'lat':legos_da['latitude']}
grid_s3 = {'lon': s3_data['longitude'], 'lat':s3_data['latitude']}
grid_smlg = {"lon": lons_sm, "lat": lats_sm}


# %%
# possibility for refactoring this later

# regridder_sit_to_nesosim = xesmf.Regridder(grid_ice_thickness, grid_nesosim, "bilinear")
# regridder_sit_to_smlg = xesmf.Regridder(grid_ice_thickness, grid_smlg, 'bilinear')
# regridder_sit_to_s3 = xesmf.Regridder(grid_ice_thickness, grid_s3, 'bilinear')
# regridder_sit_to_legos = xesmf.Regridder(grid_ice_thickness, grid_legos, 'bilinear')

# %%
grid_name_list = ['nesosim','smlg','s3','legos']
destination_grid_list = [grid_nesosim, grid_smlg, grid_s3, grid_legos]
regridder_list = [xesmf.Regridder(grid_ice_thickness, x, 'bilinear') for x in destination_grid_list]

# %%
variables_to_regrid = ['sea_ice_thickness','snow_depth','snow_density','sea_ice_density']

# %%
grid_to_variables = {}

for i,ci in enumerate(grid_name_list):
    print(ci)
    regridded_data = {}
    current_regridder = regridder_list[i]
    for varname in variables_to_regrid:
        regridded_data[varname] = current_regridder(sit_data[varname])
    grid_to_variables[ci] = regridded_data
        

# %%
grid_to_variables['nesosim']['sea_ice_thickness'][10].plot()#.plot()

# %%
nesosim_ds = xr.Dataset(grid_to_variables['nesosim']) # okay this works transparently, phew
nesosim_ds

# %%
for c in grid_name_list:
    ds_to_save = xr.Dataset(grid_to_variables[c])
    filename = 'sit_regridded_cs2/cs2_sit_on_{}_grid.nc'.format(c)
    print(filename)
    # save output with compression etc
    sc.save_xr_comp(ds_to_save, filename)
    

# %%
print('done')

# %%
grid_to_variables['nesosim']

# %%
# legos_da['snow_depth_laku_sam'][0].plot()

# %%
# plt.scatter(grid_ice_thickness['lon'], grid_ice_thickness['lat'], s=0.1, alpha=1)
# plt.show()

# %%
# sit_data['Sea_Ice_Thickness'][4].plot()

# %%
# sit_regridded_single = regridder_sit_to_nesosim(sit_data['Sea_Ice_Thickness'][4])


# %%
# sit_regridded = regridder_sit_to_nesosim(sit_data['sea_ice_thickness'])
# snod_sit_regridded = regridder_sit_to_nesosim(sit_data['snow_depth'])
# snor_sit_regridded = regridder_sit_to_nesosim(sit_data['snow_density'])
# sir_sit_regridded = regridder_sit_to_nesosim(sit_data['sea_ice_density'])

# %%
# sea water density

# density_water = 1024 # prescribed single value assumed
# delta_p = 0.9

# %%

# %%
# snow_depth_diff = ds_nesosim['snow_depth'].loc['2019-01-07'] - snod_sit_regridded

# sit_scaled_proportionally = snow_depth_diff*(density_water - snor_sit_regridded - delta_p*density_water)/(density_water - sir_sit_regridded) + sit_regridded

# %%
# sit_regridded.plot(vmin=0,vmax=4)
# plt.show()
# sit_scaled_proportionally.plot(vmin=0,vmax=4)
# plt.show()

# %%
# (sit_regridded - sit_scaled_proportionally).plot()
# plt.show()

# %%
# ((sit_regridded - sit_scaled_proportionally)/sit_regridded).plot()
# plt.show()

# %%
# ds_nesosim['snow_depth'][100].plot()

# %%
# file_path_2 = '../other_data/landy_sit/Arctic Year-Round Sea Ice Thickness Grids V3.7/uit_sit_biweekly_v2north_polarstereo_80km_20200107_v3.7.nc'

# sit_new_1file = xr.open_dataset(file_path_2, decode_times=False)
# sit_new_1file['time']

# %%
# file_path_2 = '../other_data/landy_sit/Arctic Year-Round Sea Ice Thickness Grids V3.7/uit_sit_biweekly_v2north_polarstereo_80km_20101007_v3.7.nc'

# sit_new_1file = xr.open_dataset(file_path_2, decode_times=False)
# sit_new_1file['time']

# %%
# date_start = np.datetime64('2020-01-07')
# time_diff_start = np.timedelta64(737797, 'D')
# date_start - time_diff_start

# %%
# date_start = np.datetime64('2010-10-07')
# time_diff_start = np.timedelta64(734418, 'D')
# date_start - time_diff_start

# %%
# date_initial = np.datetime64('-001-12-31')
# date_initial

# %%
# for t in sit_data['Time'].values:
#     delta_t = np.timedelta64(int(t), 'D')
#     print(date_initial + delta_t)

# %%
