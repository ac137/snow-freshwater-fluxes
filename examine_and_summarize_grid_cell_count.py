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

import fwf_helper_functions as ff
import shapefile_clipping_helpers as sc

# %%
# count corresponds to number of grid cells with snow data available

# first get count in s3_aggregated I guess?
s3_aggregated_snow_count = xr.open_mfdataset('s3_aggregated/s3_aggregated_snow_depth_count*.nc')

# %%
# s3_aggregated_snow_count['snow_depth'][0].plot() # want to clip this to fram gate and count from there I guess?

# %%
s3_depth_fram_clipped = xr.open_dataset('shapefile_clipped_data/s3_snow_depth_shapefile_clipped_Fram.nc')
s3_depth_davis_clipped = xr.open_dataset('shapefile_clipped_data/s3_snow_depth_shapefile_clipped_Davis.nc')

# %%
s3_count_in_fram = xr.where(s3_depth_fram_clipped['snow_depth']>=0,s3_aggregated_snow_count['snow_depth'],np.nan)
s3_count_in_davis = xr.where(s3_depth_davis_clipped['snow_depth']>=0,s3_aggregated_snow_count['snow_depth'],np.nan)

# %%
# s3_count_in_fram[0].plot()
# plt.xlim(-.5e6,.5e6)
# plt.ylim(-1.5e6,-.5e6)
# plt.title('snow depth count in Fram Strait single-day')
# plt.show()
# plt.colorbar(pcm,label='count')

# %%
s3_count_in_fram.loc[:'2022-04'].mean(dim=['time']).plot()
plt.xlim(-.5e6,.5e6)
plt.ylim(-1.5e6,-.5e6)
plt.title('Average count of points in Fram Strait over study period')
plt.show()

# %%
# s3_count_in_fram.loc[:'2022-04'].median(dim=['time']).plot()
# plt.xlim(-.5e6,.5e6)
# plt.ylim(-1.5e6,-.5e6)
# plt.title('Median count of points in Fram Strait over study period')
# plt.show()

# %%
s3_count_in_fram.loc[:'2022-04'].mean(dim=['time']).min().values # min over average

# %%
s3_count_in_fram.loc[:'2022-04'].median(dim=['time']).min().values # min over median

# %%
s3_count_in_fram.loc[:'2022-04'].min().values # min over all time

# %%
s3_count_in_fram.loc[:'2022-04'].mean().values # avg over all time

# %%
s3_count_in_fram.loc[:'2022-04'].plot.hist()
plt.xlabel('number of obs in grid cell')
plt.ylabel('frequency')
plt.show()

# %%
s3_count_in_davis.loc[:'2022-04'].mean(dim=['time']).plot()
plt.ylim(-1.75e6,-0.75e6)
plt.xlim(-2.5e6,-1.5e6)
plt.title('Average count of points in Davis Strait over study period')
plt.show()

# %%
s3_count_in_davis.loc[:'2022-04'].mean(dim=['time']).min().values # min over average

# %%
s3_count_in_davis.loc[:'2022-04'].plot.hist()
plt.xlabel('number of obs in grid cell')
plt.ylabel('frequency')
plt.show()

# %%
s3_count_in_davis.loc[:'2022-04'].mean().values

# %%
# Next: number of grid cells in means

s3_depth_fram_clipped['snow_depth'].count(dim=['x','y'])
s3_depth_davis_clipped['snow_depth'].count(dim=['x','y'])

# %%
dataset_list = ['nesosim','smlg-m2','amsr2','legos_sam']
fram_data_list = []
davis_data_list = []
for dataset_name in dataset_list:
    fram_data_list.append(xr.open_dataset('shapefile_clipped_data/{}_snow_depth_shapefile_clipped_Fram.nc'.format(dataset_name)))
    davis_data_list.append(xr.open_dataset('shapefile_clipped_data/{}_snow_depth_shapefile_clipped_Davis.nc'.format(dataset_name)))

# %%
fram_data_list.append(s3_depth_fram_clipped)
davis_data_list.append(s3_depth_davis_clipped)


# %%
# fram_data_list
time_index_names = ['day','time','date','time','time']

# %%
data_name_list = ['NESOSIM','SnowModel-LG','5VLSTM','LEGOS','NaRRS']
data_daily = ['NESOSIM','SnowModel-LG','5VLSTM']
data_monthly = ['LEGOS','NaRRS']

# %%
fram_daily_dict = {}
fram_monthly_dict = {}
davis_daily_dict = {}
davis_monthly_dict = {}

for i,c in enumerate(data_name_list):
    if c in data_daily:
        fram_daily_dict[c] = fram_data_list[i]['snow_depth'].count(dim=['x','y']).to_dataframe()['snow_depth']
        davis_daily_dict[c] = davis_data_list[i]['snow_depth'].count(dim=['x','y']).to_dataframe()['snow_depth']
        fram_monthly_dict[c] = fram_data_list[i]['snow_depth'].resample({time_index_names[i]:'MS'}).mean().count(dim=['x','y']).to_dataframe()['snow_depth']
        davis_monthly_dict[c] = davis_data_list[i]['snow_depth'].resample({time_index_names[i]:'MS'}).mean().count(dim=['x','y']).to_dataframe()['snow_depth']
    else:
        fram_monthly_dict[c] = fram_data_list[i]['snow_depth'].count(dim=['x','y']).to_dataframe()['snow_depth']
        davis_monthly_dict[c] = davis_data_list[i]['snow_depth'].count(dim=['x','y']).to_dataframe()['snow_depth']

# %%



# %%
def df_from_dict_date_adjust(data_dict, date_start, date_end):
    df_concat = pd.concat(data_dict,axis=1).loc[date_start:date_end]
    month_list = [1,2,3,4,10,11,12]
    df_concat = df_concat[df_concat.index.month.isin(month_list)]
    return df_concat



# %%
def exclude_summer_months(df):
    month_list = [1,2,3,4,10,11,12]
    return df[df.index.month.isin(month_list)]


# %%
date_start = '2016-10'
date_end = '2022-04'


# should also exclude summer months
fram_daily_df = df_from_dict_date_adjust(fram_daily_dict, date_start, date_end)
fram_monthly_df = df_from_dict_date_adjust(fram_monthly_dict, date_start, date_end)
davis_daily_df = df_from_dict_date_adjust(davis_daily_dict, date_start, date_end)#pd.concat(davis_daily_dict,axis=1).loc[date_start:date_end]
davis_monthly_df = df_from_dict_date_adjust(davis_monthly_dict, date_start, date_end)#pd.concat(davis_monthly_dict,axis=1).loc[date_start:date_end]


# %%
# fram_daily_df.plot(style='.')
fram_monthly_df.plot(style='.',ax=plt.gca())

# %%
fram_monthly_df

# %%
# davis_daily_df.plot(style='.')
davis_monthly_df.plot(style='.',ax=plt.gca())

# %%
davis_monthly_df


# %%
def make_minmaxmed_df(df):
    data_dict = {}
    data_dict['med'] = df.median()

    data_dict['min'] = df[df>0].min()
    data_dict['max'] = df.max()
    return pd.concat(data_dict, axis=1)


# %%
# fram_monthly_df

# %%
make_minmaxmed_df(fram_daily_df) # not using this since I don't average this way

# %%
make_minmaxmed_df(fram_monthly_df) # use this in first column (monthly number of grid cells over fram gate)

# %%
make_minmaxmed_df(davis_daily_df) # not using this

# %%
davis_monthly_df['NaRRS']['2017-11-01'] = np.nan

# %%
make_minmaxmed_df(davis_monthly_df)# use this in second column (monthly number of grid cells over davis gate)

# %%
davis_daily_df.resample('MS').sum()

# %%
fram_daily_df.resample('MS').sum()

# %%
fram_monthly_sum = exclude_summer_months(fram_daily_df.resample('MS').sum())

# %%
fram_monthly_sum['5VLSTM']['2020-10':] = np.nan # make dates after end of dataset nan so as not to skew

# %%
fram_monthly_sum['SnowModel-LG']['2021-10':] = np.nan # set missing years as nan

# %%
davis_monthly_sum = exclude_summer_months(davis_daily_df.resample('MS').sum())

# %%
davis_monthly_sum['5VLSTM']['2020-10':] = np.nan
davis_monthly_sum['SnowModel-LG']['2021-10':] = np.nan

# %%
make_minmaxmed_df(fram_monthly_sum)  #to be put in table

# %%
make_minmaxmed_df(davis_monthly_sum) # to be put in table

# %%
