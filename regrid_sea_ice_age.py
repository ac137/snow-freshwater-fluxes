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
## regrid_sea_ice_age.ipynb
# script for regridding NSIDC EASE sea ice age (Tschudi et al., 2019; https://doi.org/10.5067/UTAV7490FEPB) to snow dataset grids
# resulting interpolated values treated as myi fraction

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import os
import xesmf
import pyproj

import cartopy
import cartopy.crs as ccrs

import shapefile_clipping_helpers as sh

# %%
# load sea ice age files

# ice age reference:
# values from 0-1: first-year ice
# values from 1-16: 15-16-year-old ice
# values of 20: land-covered grid cell
# values of 21: ocean-covered grid cell

ice_age = xr.open_mfdataset('../ice_age_nsidc/*.nc/')

# %%
# nesosim and smlg files
root_path = r"../NESOSIM/output/100km"
year_file = r"final/NESOSIMv11_01092019-31082020.nc"
configstr = "ERA5CSscaledsfERA5windsNSIDCv4driftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF5.8e-07_WPT5_LLF2.9e-07-100kmv11"

sm_data_fn = "../SnowModel_data/SM_snod_ERA5_01Aug1980-31Jul2021_v01.nc"

# %%
# load nesosim and smlg
snowmodel_data = xr.open_dataset(sm_data_fn)

fn_nomelt = os.path.join(root_path, configstr,year_file)

ds_nomelt = xr.open_dataset(fn_nomelt)

# date adjustment probably not necessary but leaving in either way
ds_nomelt['day'] = pd.to_datetime(ds_nomelt['day'],format='%Y%m%d')

# %%
# LEGOS
legos_da = xr.open_mfdataset('../SD_LaKu_LEGOS_NH_Carret_and_Fleury/*.nc')

# %%
# sentinel 3 (carret & fleury)
s3_data = xr.open_dataset('s3_aggregated/s3_aggregated_snow_depth_{}.nc'.format('2018-10'))

# %%
# is2 data for mw
da_is2 = xr.open_mfdataset('../IS2SITMOGR4_v4/IS2SITMOGR4_01_*.nc')
da_is2 = da_is2.sortby('time')

# %%
# generate grid lat/lon for smlg
lons_sm, lats_sm, proj_sm, sm_grid = sh.generate_snowmodel_grid(snowmodel_data)

# %%
grid_legos = {'lon':legos_da['longitude'],'lat':legos_da['latitude']}

# %%
grid_s3 = {'lon': s3_data['longitude'], 'lat':s3_data['latitude']}

# %%
grid_ice_age = {"lon":ice_age['longitude'][0].values, "lat":ice_age['latitude'][0].values}


# %%
# drop time dimension to avoid errors about time dimension already existing when regridding
# can do this on nesosim and smlg also probably if wanting to preserve coords etc
grid_is2 = {'lon': da_is2['longitude'][0].drop('time'), 'lat':da_is2['latitude'][0].drop('time')}

# %%
# create grids and regridders for non-conservative regridding

grid_snowmodel = {"lon": lons_sm, "lat": lats_sm}
grid_nesosim = {"lon": ds_nomelt['longitude'].values, "lat": ds_nomelt['latitude'].values}

regridder_ice_age_to_nesosim = xesmf.Regridder(grid_ice_age, grid_nesosim, "bilinear")
regridder_ice_age_to_smlg = xesmf.Regridder(grid_ice_age, grid_snowmodel, "bilinear")


# %%
# regridder_ice_age_to_legos = xesmf.Regridder(grid_ice_age, grid_legos, 'bilinear')

# %%
# functions for conservative regridding

def extend_coord_array(arr):
    '''Given a 1-d array of equally spaced centre points, return an array of 
    corner points (values ranging from array[0] - array_spacing/2 
    to array[-1] + array_spacing/2)
    To be applied to arrays of x and y values in metres assuming equal spacing'''
    # get spacing between values
    arr_spacing = arr[1]-arr[0]
    # new end value = end value + spacing
    arr_end_new = arr[-1]+arr_spacing
    # create new array length len(arr)+1, last element is end value + spacing
    corner_arr = np.concat((arr,[arr_end_new]))
    # subtract half the spacing from everything, new start is start value - spacing/2, new end is end value + spacing/2
    return corner_arr - arr_spacing/2


# function to generate grid dictionaries with corner values (lon_b, lat_b) for use with conservative regridding function

def generate_corner_grid(x_coord, y_coord, projtext, grid_lat, grid_lon):
    '''Given:
    x_coord: 1-d array of x coordinate centre points (equally spaced)
    y_coord: 1-d array of y coordinate centre points (equally spaced)
    projtext: proj4text string for coordinate system corresponding to x and y
    grid_lat: latitudes corresponding to x_coord and y_coord
    grid_lon: longitudes corresponding to x_coord and y_coord
    Return a dictionary of lon (for grid square centres), lat (for grid square centres), lon_b (longitude for grid corners),
    lat_b (latitude for grid corners), ready to be passed to xesmf conservative regridders
    '''
    x_corners = extend_coord_array(x_coord)
    y_corners = extend_coord_array(y_coord)
    lons_corner, lats_corner, proj_corner, grid_corner = sh.generate_grid_params(x_corners, y_corners, projtext)
    grid_dict = {'lon':grid_lon, 'lat':grid_lat, 'lon_b':lons_corner, 'lat_b':lats_corner}
    return grid_dict

def save_to_netcdf(ds, filename):
    '''save dataset to netcdf with compression'''
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(filename, encoding=encoding)


# %%
# generate grids to be used with conservative regridders

ice_age_corner_grid = generate_corner_grid(ice_age['x'].values, ice_age['y'].values, 
                                           ice_age['crs'].attrs['proj4text'], grid_ice_age['lon'], grid_ice_age['lat'])



# %%
nesosim_corner_grid = generate_corner_grid(ds_nomelt['xgrid'][0].values, 
                                                ds_nomelt['xgrid'][0].values, 
                                                ds_nomelt['projection'].attrs['proj4text'],
                                                grid_nesosim['lat'],
                                                grid_nesosim['lon'])

smlg_proj_str = '+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +a=6371228 +b=6371228 +units=m +no_defs'

smlg_corner_grid = generate_corner_grid(snowmodel_data['x'].values, snowmodel_data['y'].values,
                                        smlg_proj_str, grid_snowmodel['lon'], grid_snowmodel['lat'])

# %%
legos_proj_str = '+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'

legos_corner_grid = generate_corner_grid(legos_da['xc'].values,legos_da['yc'].values, legos_proj_str, grid_legos['lon'],grid_legos['lat'])

# %%
s3_proj_str = '+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'
s3_corner_grid = generate_corner_grid(s3_data['x'].values,s3_data['y'].values, s3_proj_str, grid_s3['lon'],grid_s3['lat'])

# %%
is2_proj_str = da_is2['crs'].attrs['proj4text']
is2_corner_grid = generate_corner_grid(da_is2['x'].values,da_is2['y'].values, is2_proj_str, grid_is2['lon'],grid_is2['lat'])

# %%
# create conservative regridders

regridder_ice_age_to_nesosim_conservative = xesmf.Regridder(ice_age_corner_grid, nesosim_corner_grid, "conservative")
regridder_ice_age_to_smlg_conservative = xesmf.Regridder(ice_age_corner_grid, smlg_corner_grid, "conservative")

# %%
regridder_ice_age_to_legos_conservative = xesmf.Regridder(ice_age_corner_grid, legos_corner_grid, "conservative")

# %%
regridder_ice_age_to_s3_conservative = xesmf.Regridder(ice_age_corner_grid, s3_corner_grid, 'conservative')

# %%
regridder_ice_age_to_is2_conservative = xesmf.Regridder(ice_age_corner_grid, is2_corner_grid, 'conservative')

# %%
ice_age_values = ice_age['age_of_sea_ice']
ice_age_values

# %%
day_sel = 10 # date for plotting

# %%
############ creating MYI fraction mask

# create binary ice age mask with float values for reinterpolation
ice_age_values_zero_masked = xr.where(ice_age_values<20, ice_age_values, 0)
myi_binary_mask = xr.where(ice_age_values_zero_masked>1, 1., 0.)



# %%
# regrid the binary mask conservatively to obtain estimate of MYI fraction

myi_fraction_nesosim = regridder_ice_age_to_nesosim_conservative(myi_binary_mask)
myi_fraction_smlg = regridder_ice_age_to_smlg_conservative(myi_binary_mask)


# %%
myi_fraction_legos = regridder_ice_age_to_legos_conservative(myi_binary_mask)

# %%
myi_fraction_s3 = regridder_ice_age_to_s3_conservative(myi_binary_mask)

# %%
myi_fraction_is2 = regridder_ice_age_to_is2_conservative(myi_binary_mask)

# %%

# %%
myi_fraction_nesosim[day_sel].plot()
plt.show()

myi_fraction_smlg[day_sel].plot()
plt.show()
# seems reasonable

# %%
myi_fraction_legos[day_sel].plot()
plt.show()

# %%
myi_fraction_s3[day_sel].plot()
plt.show()

# %%
myi_fraction_is2[day_sel].plot()
plt.show()

# %%



# %%
# calculate monthly average of these datasets
myi_fraction_nesosim_monthly = myi_fraction_nesosim.resample(time="1MS").mean()
myi_fraction_smlg_monthly = myi_fraction_smlg.resample(time="1MS").mean()




# %%
# assign name attribute and save

myi_fraction_nesosim_monthly = myi_fraction_nesosim_monthly.to_dataset(name='myi_fraction')

myi_fraction_smlg_monthly = myi_fraction_smlg_monthly.to_dataset(name='myi_fraction')

# %%
myi_fraction_legos_monthly = myi_fraction_legos.resample(time='1MS').mean()

myi_fraction_legos_monthly = myi_fraction_legos_monthly.to_dataset(name='myi_fraction')

# %%
myi_fraction_s3_monthly = myi_fraction_s3.resample(time='1MS').mean()
myi_fraction_s3_monthly = myi_fraction_s3_monthly.to_dataset(name='myi_fraction')

# %%
myi_fraction_is2_monthly = myi_fraction_is2.resample(time='1MS').mean()
myi_fraction_is2_monthly = myi_fraction_is2_monthly.to_dataset(name='myi_fraction')

# %%
# save myi fraction to netcdf with compression

save_to_netcdf(myi_fraction_nesosim_monthly, 'monthly_myi_fraction_nesosim_grid.nc')

save_to_netcdf(myi_fraction_smlg_monthly, 'monthly_myi_fraction_SMLG_grid.nc')



# %%
save_to_netcdf(myi_fraction_legos_monthly, 'monthly_myi_fraction_legos_grid.nc')

# %%
save_to_netcdf(myi_fraction_s3_monthly, 'monthly_myi_fraction_s3_grid.nc')

# %%
save_to_netcdf(myi_fraction_is2_monthly, 'monthly_myi_fraction_is2_grid.nc')

# %%
'done'

# %%
############## creating binary MYI mask; not used for publication but included for reference

# start with zeroed out land values

# ice_age_values_zero_masked

# then regrid to corresponding grids; may as well use conservative regridder
ice_age_nesosim = regridder_ice_age_to_nesosim_conservative(ice_age_values_zero_masked.astype(np.float64))

ice_age_smlg = regridder_ice_age_to_smlg_conservative(ice_age_values_zero_masked.astype(np.float64))



# %%
# plot original and regridded values
ice_age_values_zero_masked[day_sel].plot()
plt.show()

ice_age_smlg[day_sel].plot()
plt.show()

ice_age_nesosim[day_sel].plot()
plt.show()


# %%
# once regridded, calculate monthly average

ice_age_nesosim_monthly = ice_age_nesosim.resample(time="1MS").max()

ice_age_smlg_monthly = ice_age_smlg.resample(time="1MS").max()


# %%
# generate a binary mask from the monthly average; if >1 set to MYI, otherwise set to FYI

myi_binary_mask_nesosim_monthly = xr.where(ice_age_nesosim_monthly>1.5, 1, 0)

myi_binary_mask_smlg_monthly = xr.where(ice_age_smlg_monthly>1.5, 1, 0)

# %%


myi_binary_mask.resample(time="1MS").max()[day_sel//4].plot()
plt.show()

myi_binary_mask_smlg_monthly[day_sel//4].plot()
plt.show()

myi_binary_mask_nesosim_monthly[day_sel//4].plot()
plt.show()
