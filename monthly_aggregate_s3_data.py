#############################################################################
# script for monthly aggregation of S3 NaRRS data
# NaRRS data from https://zenodo.org/records/17942492
# cf Nelson et al. 2026 (GRL) and https://github.com/ConnorAJNelson/NaRRS

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pyproj
import os
import satpy
import pyresample
# from pyresample.bucket import BucketResampler
# from pyresample.geometry import SwathDefinition

# dataset for destination grid (from NaRRS repository)
s3_monthly = xr.open_dataset('../s3_snow_data/Gridded_S3_NaRRS_snow_depth_and_sea_ice_thickness_nh_202110.nc')

data_dir_s3a = '../s3_snow_data/*/'
data_dir_s3b = '../s3_snow_data/S3B/'
file_name = 'NaRRS_Sentinel3_snow_depth_and_sea_ice_thickness.nc'

year_range = ['2018', '2019','2020','2021','2022','2023']
# year_range = ['2018', '2019']
# probably  don't need to hardcode all this but just leaving as-is for now
# skipping may-september because those dates are absent from the data
month_range = ['01','02','03','04','10','11','12']
# month_range = ['10','11','12']
# month_range = ['01','02','03','04']

# area definition from the destination grid dataset
area_def = pyresample.geometry.AreaDefinition('EASE', 'NSIDC EASE-Grid 2.0', 'EASE', 
									  '+datum=WGS84 +lat_0=90 +lon_0=0 +no_defs +proj=laea +type=crs +units=m +x_0=0 +y_0=0',
					   432, 432,
					   [s3_monthly['x'][0].values, s3_monthly['y'][0].values,
						s3_monthly['x'][-1].values, s3_monthly['y'][-1].values])

for y in year_range:
	for m in month_range:
		# data availability flags
		s3a_data_available = True
		s3b_data_available = True
		
		year_month_current = '{}{}'.format(y,m)
		year_month_current_with_delim = '{}-{}'.format(y,m)
		# catchall to select all filenames with corresponding dates in the s3 directory
		file_name_batch_a = data_dir_s3a + '*/S3A_SR_2_LAN_SI*_{}*_2023*/'.format(year_month_current) + file_name
		file_name_batch_b = data_dir_s3b + '*/S3B_SR_2_LAN_SI*_{}*_2023*/'.format(year_month_current) + file_name

		print(year_month_current_with_delim)
		print('loading swath data')

		# load swath data
		# try loading from both s3a and s3b
		try:
			# try loading s3a
			s3_data_a = xr.open_mfdataset(file_name_batch_a)
		except OSError:
			print('no s3a data')
			s3_data_a = None
			s3a_data_available = False
		try: 
			# try loading s3b
			s3_data_b = xr.open_mfdataset(file_name_batch_b)
		except OSError:
			print('no s3b data')
			s3_data_b = None
			s3b_data_available = False
		# cases: both data are available
		if s3a_data_available and s3b_data_available:
			print('loading both s3a and s3b')
			# concatenate the datasets
			lon_arr_a = s3_data_a['lon_01'].loc[year_month_current_with_delim].values
			lon_arr_b = s3_data_b['lon_01'].loc[year_month_current_with_delim].values
			lat_arr_a = s3_data_a['lat_01'].loc[year_month_current_with_delim].values
			lat_arr_b = s3_data_b['lat_01'].loc[year_month_current_with_delim].values

			lons_swath = xr.DataArray(np.array([np.concat([lon_arr_a, lon_arr_b])]), dims=('x', 'y'))
			lats_swath = xr.DataArray(np.array([np.concat([lat_arr_a, lat_arr_b])]), dims=('x', 'y'))

			s3_sd_a = s3_data_a['snow_depth_01'].loc[year_month_current_with_delim].values
			s3_sd_b = s3_data_b['snow_depth_01'].loc[year_month_current_with_delim].values
			sd_arr_all = np.array([np.concat([s3_sd_a, s3_sd_b])])
			start_time = s3_data_a['time_01'][0].values
			print('start time :',start_time) # not strictly accurate but close enough
			print('end time :',s3_data_b['time_01'][-1].values)
		elif (not s3a_data_available) and (not s3b_data_available):
			# neither dataset available
			print('no data available for '+ year_month_current_with_delim)
			continue
		else:
			# only one of the datasets available 
			if s3a_data_available:
				print('loading s3a')
				s3_data = s3_data_a
			elif s3b_data_available:
				print('loading s3b')
				s3_data = s3_data_b
			# 
			lons_swath = xr.DataArray(np.array([s3_data['lon_01'].loc[year_month_current_with_delim].values]), dims=('x', 'y'))
			lats_swath = xr.DataArray(np.array([s3_data['lat_01'].loc[year_month_current_with_delim].values]), dims=('x', 'y'))
			sd_arr_all = np.array([s3_data['snow_depth_01'].loc[year_month_current_with_delim].values])
			start_time = s3_data['time_01'][0].values

			print('start time :',start_time)
			print('end time :',s3_data['time_01'][-1].values)

		# area definition for source (swath) data
		area_def1 = pyresample.geometry.SwathDefinition(lons=lons_swath,
														lats=lats_swath)

		print('creating scene')
		# create a scene object from the swath data
		scene = satpy.Scene()
		# assign snow depth in scene, assign area definitions
		# need to assign x/y dimensions for the regridding to work even though this is 1d data
		scene['snow_depth'] = xr.DataArray(
		    # subset selection for specific year-month
		    data=sd_arr_all,
		    dims=('x', 'y'),
		    attrs={'name': 'test',
		           'start_time': start_time,
		           'area': area_def1})
		print('resampling to destination grid and formatting data')
		# resample scene to destination area and get bucket average
		scene_resampled = scene.resample(area_def, resampler='bucket_avg')
		# get bucket count
		scene_resampled_count = scene.resample(area_def,resampler='bucket_count')

		xr_resampled = scene_resampled.to_xarray()
		xr_count_resampled = scene_resampled_count.to_xarray()
		time_value = pd.to_datetime('{}-01'.format(year_month_current_with_delim))
		print('saving data')
		xr_resampled.expand_dims(time=[time_value]).to_netcdf('s3_aggregated/s3_aggregated_snow_depth_{}.nc'.format(year_month_current_with_delim))
		xr_count_resampled.expand_dims(time=[time_value]).to_netcdf('s3_aggregated/s3_aggregated_snow_depth_count_{}.nc'.format(year_month_current_with_delim))
		print('saved')
		# break # for testing
	# break # for testing

