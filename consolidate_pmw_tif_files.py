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
##########################################################################################################
# consolidate_pmw_tif_files.ipynb
# script to consolidate amsr2 passive microwave snow depth tiff files (Li et al 2022) onto a common grid
# and output as netcdf
# also regrids to NESOSIM and SnowModel-LG grids
###########################################################################################################

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

import xesmf
import shapefile_clipping_helpers as sc

# %%
# load rioxarray data

# year_current = 2019
# path_current = './{}/N{}01*.tif'.format(year_current,year_current)

# path_current = './{}/N{}0101_LSTM_hs.tif'.format(year_current,year_current)

# %%

# data_paths = [os.path.join(year_path, f) for f in data_files]
# data_paths

# %%
# data_files[0][1:9]

# %%
data_list = []

for year_current in range(2012, 2021):
# for year_current in range(2018, 2019): # for plotting single files only
    print(year_current)
    # directory where data is stored; change as needed
    data_dir = '.'
    # year_current = 2019
    
    year_path = os.path.join(data_dir,str(year_current))
    data_files = os.listdir(year_path)
    
    print('loading_files')
    for p in data_files:
        date_str = p[1:9]
        date_val = pd.to_datetime(date_str, format='%Y%m%d')
        #assign date coordinate and stack. 
        pmw_data = rio.open_rasterio(os.path.join(year_path, p)).assign_coords({'date':date_val}).expand_dims({'date': 1})
         
        data_list.append(pmw_data)

# %%
# test loading a single data file for reference; adjust path as needed
path_single = './2018/N20181201_LSTM_hs.tif'
single_data_file = rio.open_rasterio(path_single).assign_coords({'date':'2018-12-01'}).expand_dims({'date': 1})

# %%
single_data_file[0,0]

# %%
# single-file projections
proj_pmw_single = pyproj.crs.CRS.from_wkt(single_data_file['spatial_ref'].attrs['crs_wkt'])
 # okay I could get the projection
proj_pmw_single = pyproj.proj.Proj(proj_pmw_single.to_proj4())
xs, ys = np.meshgrid(single_data_file.x,single_data_file.y)
lons, lats = proj_pmw_single(xs,ys,inverse=True)

# %%
# plot the single file



proj=ccrs.NorthPolarStereo(central_longitude=-45)
proj_coord = ccrs.PlateCarree()


fig, ax = plt.subplots(1,1,dpi=200,figsize=(6,6),subplot_kw=dict(projection=proj))
ax.set_facecolor('#dddddd')
# plot lat/lon marked with points
plt.title('5VLSTM {}'.format('2018-12-01'))
# plt.plot(data['Longitude'],data['Latitude'],lw=0,marker='.',transform=proj_coord)
# or for pcolormesh
plt.pcolormesh(lons,lats, single_data_file[0,0]/100, transform=proj_coord,cmap='Blues',alpha=1,vmin=0,vmax=0.7)
vmax=0.4

# plt.pcolormesh(lons_sm,lats_sm, clipped_smlg[0], transform=proj_coord,vmin=0,vmax=vmax)
# plt.pcolormesh(lons_sm,lats_sm, clipped_smlg[1], transform=proj_coord,vmin=0,vmax=vmax)
shape_alpha = 0.3
# shapefiles_smlg[0].plot(ax=ax,color='red',alpha=shape_alpha,transform=crs_smlg)
# shapefiles_smlg[1].plot(ax=ax,color='red',alpha=shape_alpha,transform=crs_smlg)
ax.coastlines()#resolution='110m') # specify coastline resolution
ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree()) # specify axis extent
plt.colorbar(label='Snow depth (m)') # colorbar for pcolormesh
plt.show()

# %%
data_list_unified = []

# select subsets of each dataset to unify all data with the same coordinates

for i, item in enumerate(data_list):

    ############ code to round the coordinates to unify individual files on the same grid
    item['x'] = np.round(item['x']/5000) * 5000
    item['y'] = np.round(item['y']/5000) * 5000

    data_list_unified.append(item)


# %%
xy_bound_list = []
date_list = []
proj_string_list = []

for i, item in enumerate(data_list):
    xy_bound_list.append( np.array([item.x[0],item.y[0],item.x[-1],item.y[-1]]))
    date_list.append(item.date.values[0])
    proj_string_list.append(item['spatial_ref'].attrs['crs_wkt'])

xy_bound_stack = np.vstack(xy_bound_list)

df_bounds = pd.DataFrame(xy_bound_stack, index=date_list,columns=['x0','y0','xf','yf'])

display((df_bounds.div(5000).round(0) * 5000).drop_duplicates())

display((df_bounds - (df_bounds.div(5000).round(0) * 5000)).min())

# %%
# concatenate all the data together
# note: variable is called pmw_data_1_year but it will contain as many years as available
pmw_data_1year = xr.concat(data_list_unified,dim='date')

# sort the data in order of date
pmw_data_1year = pmw_data_1year.sortby('date')
pmw_data_1year_ds = pmw_data_1year.to_dataset(name='snow_depth') # create dataset

# %%
# save the consolidated data as a netcdf
sc.save_xr_comp(pmw_data_1year_ds, './li_amsr2_pmw/pmw_snow_depth_2012_to_2020.nc')


# %%
# load nesosim & smlg data for grid setup; adjust file paths as needed

nesosim_data = xr.open_dataset('../NESOSIM/output/100km/ERA5CSscaledsfERA5windsNSIDCv4driftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF2.0504155592128743e-06_WPT5_LLF4.0059442776163867e-07-100kmv11/final/NESOSIMv11_01092019-31082020.nc')
sm_data = xr.open_dataset('../SnowModel_data/SM_snod_ERA5_01Aug1980-31Jul2021_v01.nc')


# %%
# set up regridders for data

# get coords from aggregated file
proj_pmw = pyproj.crs.CRS.from_wkt(pmw_data_1year['spatial_ref'].attrs['crs_wkt'])
 # okay I could get the projection
proj_pmw = pyproj.proj.Proj(proj_pmw.to_proj4())
x, y = np.meshgrid(pmw_data_1year.x,pmw_data_1year.y)
lon, lat = proj_pmw(x,y,inverse=True)
# pmw lon/lat


grid_dict_pmw = {'lon':lon,'lat':lat}
grid_dict_nesosim = {"lon": nesosim_data['longitude'].values, "lat": nesosim_data['latitude'].values}
lons_sm, lats_sm, proj_sm, grid_sm = sc.generate_snowmodel_grid(sm_data)

grid_dict_smlg = {"lon": lons_sm, 'lat':lats_sm}


# regridder setup
regridder_pmw_to_nesosim = xesmf.Regridder(grid_dict_pmw, grid_dict_nesosim, "bilinear")
regridder_pmw_to_snowmodel = xesmf.Regridder(grid_dict_pmw, grid_dict_smlg, "bilinear")


# %%
# regrid data to nesosim and smlg grids
pmw_on_nesosim_grid = regridder_pmw_to_nesosim(pmw_data_1year_ds)
pmw_on_smlg_grid = regridder_pmw_to_snowmodel(pmw_data_1year_ds)

# assign x and y values to snowmodel data
pmw_on_smlg_grid['x'] = sm_data['x']

pmw_on_smlg_grid['y'] = sm_data['y']

# %%
# mask out the spurious zero values at domain edges from regridding
# note: this assumes that data[0] has no zero values elsewhere (which is true currently but may not always be the case)

pmw_on_nesosim_grid['snow_depth'] = xr.where((pmw_on_nesosim_grid['snow_depth'][0]!=0).values,pmw_on_nesosim_grid['snow_depth'],np.nan)

pmw_on_smlg_grid['snow_depth'] = xr.where((pmw_on_smlg_grid['snow_depth'][0]!=0).values,pmw_on_smlg_grid['snow_depth'],np.nan)


# %%
# save files

sc.save_xr_comp(pmw_on_nesosim_grid, 'pmw_amsr2_li_nesosim_grid.nc')
sc.save_xr_comp(pmw_on_smlg_grid, 'pmw_amsr2_li_smlg_grid.nc')
print('files saved')

# %%
# example plots to ensure orientation is correct
date_chosen = '2013-03-01'

pmw_on_nesosim_grid['snow_depth'].loc[date_chosen].plot()
plt.show()
pmw_on_smlg_grid['snow_depth'].loc[date_chosen].plot()
plt.show()

# %%
(pmw_on_nesosim_grid['snow_depth'][0]==0).plot() # double-check the zero edges from regridding
plt.show()

# %%
# test loading data
# data_load_test = xr.open_dataset('./li_amsr2_pmw/pmw_snow_depth_year_{}.nc'.format(year_current))
# data_load_test

# %%
# data_load_test['spatial_ref'].attrs

# %%
# plot data as example by date
pmw_data_1year.loc['2020-04-01'].plot()
plt.show()

# %%
idx_to_plot = 400

data_current = data_list[idx_to_plot]
# plot the data on a map
# doing this separately is somewhat redundant because each tiff does have the same spatial ref
proj_pmw2 = pyproj.crs.CRS.from_wkt(data_current['spatial_ref'].attrs['crs_wkt'])
proj_pmw2 = pyproj.proj.Proj(proj_pmw2.to_proj4())
x2, y2 = np.meshgrid(data_current.x,data_current.y)
lon2, lat2 = proj_pmw2(x2,y2,inverse=True)

date_current = data_current.date.values
print(date_current)
fig, ax = plt.subplots(1,1,dpi=200,figsize=(4,4),subplot_kw=dict(projection=proj))

# plot lat/lon marked with points

# plt.plot(data['Longitude'],data['Latitude'],lw=0,marker='.',transform=proj_coord)
# or for pcolormesh
plt.pcolormesh(lon2, lat2, data_current[0,0], transform=proj_coord)
ax.coastlines(resolution='110m') # specify coastline resolution
ax.set_extent([-180, 180, 65, 90], ccrs.PlateCarree()) # specify axis extent
# plt.colorbar() # colorbar for pcolormesh
plt.show()

# %%
# plot the data on a map
# date_current = '2019-01-01'
# date_current = '2012-10-31'

proj=ccrs.NorthPolarStereo(central_longitude=-90)
proj_coord = ccrs.PlateCarree()
fig, ax = plt.subplots(1,1,dpi=200,figsize=(4,4),subplot_kw=dict(projection=proj))

# plot lat/lon marked with points

# plt.plot(data['Longitude'],data['Latitude'],lw=0,marker='.',transform=proj_coord)
# or for pcolormesh
plt.pcolormesh(lon2, lat2, data_current[0,0], transform=proj_coord,cmap='Blues')

plt.pcolormesh(lon, lat,pmw_data_1year.loc[date_current][0,0], transform=proj_coord,cmap='Reds')

ax.coastlines(resolution='110m') # specify coastline resolution
ax.set_extent([-180, 180, 65, 90], ccrs.PlateCarree()) # specify axis extent
# plt.colorbar() # colorbar for pcolormesh
plt.show()

# %%
proj=ccrs.NorthPolarStereo(central_longitude=-90)
proj_coord = ccrs.PlateCarree()
fig, ax = plt.subplots(1,1,dpi=200,figsize=(4,4),subplot_kw=dict(projection=proj))

# plot lat/lon marked with points

# plt.plot(data['Longitude'],data['Latitude'],lw=0,marker='.',transform=proj_coord)
# or for pcolormesh
plt.pcolormesh(lon, lat,pmw_data_1year.loc[date_current][0,0], transform=proj_coord)
ax.coastlines(resolution='110m') # specify coastline resolution
ax.set_extent([-180, 180, 65, 90], ccrs.PlateCarree()) # specify axis extent
# plt.colorbar() # colorbar for pcolormesh
plt.show()

# %%
