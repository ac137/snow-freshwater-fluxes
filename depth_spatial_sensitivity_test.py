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
import fwf_helper_functions as ff

# %%
# load clipped data
product_name = 'smlg-m2'
# product_name = 'legos_sam'
# product_name = 's3'
# product_name = 'amsr2'
# product_name = 'nesosim'
line_colours = ['#66CCEE','#4477aa','#228833','#CCBB44','#AA3377','#BBBBBB','#EE6677']
if product_name in ['smlg-m2','legos_sam','s3']:
    time_var = 'time'
elif product_name == 'amsr2':
    time_var = 'date'
elif product_name == 'nesosim':
    time_var = 'day'

# look at smlg-m2 first
depth_fram_fn = 'shapefile_clipped_data/{}_snow_depth_shapefile_clipped_Fram.nc'.format(product_name)
depth_fn_davis = 'shapefile_clipped_data/{}_snow_depth_shapefile_clipped_Davis.nc'.format(product_name)


data_clipped_fram = xr.open_dataset(depth_fram_fn)


# %%
def apply_mask_to_salinity(snow_salinity, myi_mask):
    '''scale salinity by multiyear ice fraction'''
    return snow_salinity*(1-myi_mask.values)

def calc_salinity_height(snow_height, snow_salinity, snow_density):
    '''calculate snow freshwater water-equivalent height by scaling by salinity and density
    uses prescribed reference salinity (sal_ref) and assumed seawater density (dens_seawater)'''
    sal_ref = 34.8
    dens_seawater = 1023.9
    return snow_height*(1-snow_salinity/sal_ref)*(snow_density/dens_seawater)

def get_freshwater_quantiles(product_name):
    if product_name in ['smlg-m2','legos_sam','s3']:
        time_var = 'time'
    elif product_name == 'amsr2':
        time_var = 'date'
    elif product_name == 'nesosim':
        time_var = 'day'

    fram_ice_flux_data = ff.load_and_preprocess_ice_flux_data('fram-strait-ice-fluxes-v2.csv')
    davis_ice_flux_data = ff.load_and_preprocess_ice_flux_data('davis-strait-ice-fluxes-v2.csv')

    
    depth_quartiles = {}
    fwf_estimates = {}

    for strait in ['Fram','Davis']:

        depth_fn = 'shapefile_clipped_data/{}_snow_depth_shapefile_clipped_{}.nc'.format(product_name,strait)
        # depth_fn_davis = 'shapefile_clipped_data/{}_snow_depth_shapefile_clipped_Davis.nc'.format(product_name)
        
        
        data_clipped = xr.open_dataset(depth_fn)
        # data_clipped_davis = xr.open_dataset(depth_davis_fn)
    
        q1_depths = data_clipped['snow_depth'].quantile(0.25,dim=['x','y'])
        q2_depths = data_clipped['snow_depth'].quantile(0.5,dim=['x','y'])
        q3_depths = data_clipped['snow_depth'].quantile(0.75,dim=['x','y'])
        depth_arrs = [q1_depths, q2_depths, q3_depths]
        depth_arrs_monthly = [x.resample({time_var:'MS'}).mean() for x in depth_arrs] 
        depth_arrs_monthly = [x.to_dataframe() for x in depth_arrs_monthly]

        depth_quartiles[strait] = depth_arrs_monthly

        snow_density = xr.open_dataset('shapefile_clipped_data/{}_snow_density_shapefile_clipped_{}.nc'.format(product_name,strait))
        snow_salinity = xr.open_dataset('shapefile_clipped_data/{}_snow_salinity_shapefile_clipped_{}.nc'.format(product_name,strait))
        myi_fraction = xr.open_dataset('shapefile_clipped_data/{}_myi_fraction_shapefile_clipped_{}.nc'.format(product_name,strait))

        snow_salinity2 = 0 # assume zero snow salinity for simplicity
        snow_density2 = 300 # assume fixed snow density for simplicity

        sal_height_estimate_different_depths = [calc_salinity_height(x['snow_depth'], snow_salinity2, snow_density2) for x in depth_arrs_monthly]
        fwf_estimate_different_depths = [fram_ice_flux_data['IAF']*x/1000 for x in sal_height_estimate_different_depths]

        fwf_estimates[strait] = fwf_estimate_different_depths
    return depth_quartiles, fwf_estimates

# %%
product_list = ['nesosim','smlg-m2','amsr2','s3','legos_sam']
product_clean_names = ['NESOSIM','SnowModel-LG','5VLSTM','NaRRS','LEGOS']

product_clean_names_plot = ['NESOSIM','NESOSIM','SnowModel-LG','SnowModel-LG','5VLSTM','5VLSTM','NaRRS','NaRRS','LEGOS','LEGOS']

# %%
depth_arrs_all = {}
fwf_estimates_all = {}

for product_name in ['nesosim','smlg-m2','amsr2','s3','legos_sam']:
    print(product_name)
    depth_arrs_all[product_name], fwf_estimates_all[product_name] = get_freshwater_quantiles(product_name)

# depth_arrs_smlg, fwf_estimates_smlg = get_freshwater_quantiles('smlg-m2')
# depth_arrs_nesosim, fwf_estimates_nesosim = get_freshwater_quantiles('smlg-m2')

# %%
def product_df_from_lists(data_all):
    product_dict_1 = {}
    for i, pn in enumerate(product_list):
        gate_dict = {}
        for j,strait in enumerate(['Fram','Davis']):
            if pn in ['smlg-m2','legos_sam','s3']:
                time_var = 'time'
            elif pn == 'amsr2':
                time_var = 'date'
            elif pn == 'nesosim':
                time_var = 'day'
            gate_dict[strait] = pd.concat(data_all[pn][strait]).reset_index().rename(columns={time_var:'date'}).pivot(index='date', columns='quantile', values='snow_depth')
        product_dict_1[product_clean_names[i]] = pd.concat(gate_dict,axis=1)#
        # break
    # break
    
    return pd.concat(product_dict_1,axis=1)


# %%
# create a dataframe
product_dict_1 = {}
for i, pn in enumerate(product_list):
    gate_dict = {}
    for j,strait in enumerate(['Fram','Davis']):
        if pn in ['smlg-m2','legos_sam','s3']:
            time_var = 'time'
        elif pn == 'amsr2':
            time_var = 'date'
        elif pn == 'nesosim':
            time_var = 'day'
        gate_dict[strait] = pd.concat(depth_arrs_all[pn][strait]).reset_index().rename(columns={time_var:'date'}).pivot(index='date', columns='quantile', values='snow_depth')
    product_dict_1[product_clean_names[i]] = pd.concat(gate_dict,axis=1)#
    # break
# break

product_df_all = pd.concat(product_dict_1,axis=1)

# product_dict_1

# %%
product_df_all_depth = product_df_from_lists(depth_arrs_all)

# %%
fwf_estimates_all['nesosim']['Fram'][0]

product_dict_1 = {}
for i, pn in enumerate(product_list):
    gate_dict = {}
    for j,strait in enumerate(['Fram','Davis']):
        if pn in ['smlg-m2','legos_sam','s3']:
            time_var = 'time'
        elif pn == 'amsr2':
            time_var = 'date'
        elif pn == 'nesosim':
            time_var = 'day'
        # print(fwf_estimates_all[pn][strait])
    #     break
    # break
        gate_dict[strait] = pd.concat(fwf_estimates_all[pn][strait],axis=1).rename(columns={0:'q1',1:'q2',2:'q3'})#pd.concat(depth_arrs_all[pn][strait]).reset_index().rename(columns={time_var:'date'}).pivot(index='date', columns='quantile', values='snow_depth')
    product_dict_1[product_clean_names[i]] = pd.concat(gate_dict,axis=1)#
#     # break
# # break

product_df_all_fw = pd.concat(product_dict_1,axis=1)

# %%
product_df_all_fw

# %%
# product_df_all_fwf = product_df_from_lists(fwf_estimates_all)

# %%
fig,ax = plt.subplots(2,1,dpi=300,figsize=(10,10))
# color_list = 
for i, pn in enumerate(product_list):
    for j,strait in enumerate(['Fram','Davis']):
        fwf_current = fwf_estimates_all[pn][strait]
        # print(fwf_current)
        fwf_current[1].loc[:'2022-04'].plot(ax=ax[j],color=line_colours[i])
        ax[j].fill_between(fwf_current[1].loc[:'2022-04'].index,fwf_current[0].loc[:'2022-04'],fwf_current[2].loc[:'2022-04'],color=line_colours[i],alpha=0.2)
        # for k in range(len(fwf_current)):
        #     fwf_current[k].plot(ax=ax[j],color=line_colours[i])
        # break
    # break
ax[0].set_ylabel('Snow freshwater flux (km$^3$)')
ax[1].set_ylabel('Snow freshwater flux (km$^3$)')
plt.suptitle('Snow freshwater flux spatial variability; median, q1, q3')
ax[0].set_title('Fram Strait')
ax[1].set_title('Davis Strait')
plt.tight_layout()
ax[0].legend(labels=product_clean_names_plot,ncol=2)
plt.show()

# %%
month_list = [10,11,12,1,2,3,4]
products_winter = product_df_all[product_df_all.index.month.isin(month_list)]

# %%
products_winter

# %%
products_winter.resample('YS-OCT').mean()#.mean()

# %%
product_df_all_fw_winter = product_df_all_fw[product_df_all_fw.index.month.isin(month_list)]

# %%
fwf_ranges_winter = product_df_all_fw_winter.resample('YS-OCT').mean().mean()

# %%
fwf_ranges_winter.loc[(slice(None),slice(None),'q2')]

# %%
iqr_fwf = fwf_ranges_winter.loc[(slice(None),slice(None),'q3')] - fwf_ranges_winter.loc[(slice(None),slice(None),'q1')]
# IQR

# %%
iqr_fwf/2

# %%
fwf_ranges_winter.loc[(slice(None),slice(None),'q3')] - fwf_ranges_winter.loc[(slice(None),slice(None),'q2')] # plus

# %%
fwf_ranges_winter.loc[(slice(None),slice(None),'q2')] - fwf_ranges_winter.loc[(slice(None),slice(None),'q1')] # minus

# %%
100*iqr_fwf/2/fwf_ranges_winter.loc[(slice(None),slice(None),'q2')]

# %%
# data_clipped_fram['snow_depth'][20].plot()#[100].plot()
# # plt.ylim(-2e6,0)
# # plt.xlim(-1e6,1e6)
# plt.xlim(-1e6,1e6)
# plt.ylim(-2e6,0)
# plt.show()

data_clipped_fram['snow_depth'][100].plot()#[100].plot()

plt.xlim(-1e6,1e6)
plt.ylim(-2e6,0)
plt.show()

# %%
data_clipped_fram['snow_depth'][100].plot.hist()

plt.show()

# %%
data_clipped_fram['snow_depth'].loc['2019-01'].plot.hist()

plt.show()

# %%
# min_depths = data_clipped_fram['snow_depth'].min(dim=['x','y'])
# max_depths = data_clipped_fram['snow_depth'].max(dim=['x','y'])

q1_depths = data_clipped_fram['snow_depth'].quantile(0.25,dim=['x','y'])
q2_depths = data_clipped_fram['snow_depth'].quantile(0.5,dim=['x','y'])
q3_depths = data_clipped_fram['snow_depth'].quantile(0.75,dim=['x','y'])
# or should I do the quantiles grouped by time also?
# these data are extremely non-gaussian

# %%
mean_depths = data_clipped_fram['snow_depth'].mean(dim=['x','y'])

# %%
depth_arrs = [q1_depths, q2_depths, q3_depths]

# %%
# min_depths.plot()
q1_depths.plot()
q2_depths.plot()
q3_depths.plot()
mean_depths.plot()
# max_depths.plot()
plt.show()

# %%
# resample these monthly

depth_arrs_monthly = [x.resample({time_var:'MS'}).mean() for x in depth_arrs] 


# %%


# depth_arrs_monthly_2 = [#data_clipped_fram['snow_depth'].resample(time='MS').min(dim=['time','x','y']),
#                         data_clipped_fram['snow_depth'].resample(time='MS').quantile(0.25,dim=['time','x','y']),
#                         data_clipped_fram['snow_depth'].resample(time='MS').quantile(0.5,dim=['time','x','y']),

#                        data_clipped_fram['snow_depth'].resample(time='MS').quantile(0.75,dim=['time','x','y'])]
#                        #data_clipped_fram['snow_depth'].resample(time='MS').max(dim=['time','x','y'])]

# %%
labels=['q1','q2','q3']

for i,d in enumerate(depth_arrs_monthly):
    d.plot(label=labels[i])
    # depth_arrs_monthly_2[i].plot(label=labels[i]+'_new') # okay yeah the quantile is pretty similar
plt.legend()
plt.show()

# %%
# okay quartiles are very similar, min/max just impacted (min goes to zero as expected so maybe just use depth_arrs_monthly instead)
# min/max not as informative since those values are relatively unlikely

# %%
# try scaling the fluxes by the min and the max to look at possible variability due to snow distribution?

# %%
# load fram fluxes

fram_ice_flux_data = ff.load_and_preprocess_ice_flux_data('fram-strait-ice-fluxes-v2.csv')


# %%
# fram_ice_flux_data['IAF']

# %%
depth_arrs_monthly = [x.to_dataframe() for x in depth_arrs_monthly]

# %%
# snow_depth_arrs_monthly[0]['snow_depth']*fram_ice_flux_data['IAF']

# %%
# need snow density salinity etc also, can load those files up?

snow_density = xr.open_dataset('shapefile_clipped_data/{}_snow_density_shapefile_clipped_Fram.nc'.format(product_name))
snow_salinity = xr.open_dataset('shapefile_clipped_data/{}_snow_salinity_shapefile_clipped_Fram.nc'.format(product_name))
myi_fraction = xr.open_dataset('shapefile_clipped_data/{}_myi_fraction_shapefile_clipped_Fram.nc'.format(product_name))

# %%
# snow_density # do I do regional mean or monthly mean first? I think regional mean

# %%

# %%
# need to forward fill myi mask first
# if product_name in ['nesosim','smlg-m2', 'amsr2']:
#     myi_current = myi_fraction['myi_fraction'].resample(time="1D").ffill()
# else:
#     myi_current = myi_fraction['myi_fraction']

# %%
# salinity_masked = apply_mask_to_salinity(snow_salinity['snow_salinity'],myi_current)

# %%
# hopefully they have the same dates? if not need to adjust

# %%
# try for a single year at least
# snow_salinity['snow_salinity'].loc['2019-09':'2020-08']

# %%
# myi_current.loc['2019-09':'2020-08']

# %%
# salinity_masked = apply_mask_to_salinity(snow_salinity['snow_salinity'].loc['2019-09':'2020-08'],
#                                          myi_current.loc['2019-09':'2020-08'])

# %%
# have masked salinity and snow height and density
# actually what if we just keep it simple
# depth_arrs_monthly

# %%
# depth_arrs_monthly[0]

# %%
snow_salinity2 = 0 
snow_density2 = 300

sal_height_estimate_different_depths = [calc_salinity_height(x['snow_depth'], snow_salinity2, snow_density2) for x in depth_arrs_monthly]

# %%
# sal_height_estimate_different_depths

# %%
fwf_estimate_different_depths = [fram_ice_flux_data['IAF']*x/1000 for x in sal_height_estimate_different_depths]

# %%

for i,d in enumerate(fwf_estimate_different_depths):
    d.plot(label=labels[i])
plt.legend()
plt.title('Estimated snow freshwater fluxes using different depths from distribution')
plt.show()

# %%
# distribution of snow water equivalent

# swe = snow_depth*snow_density/water_density (excluding water density term here)
swe_total = data_clipped_fram['snow_depth']*snow_density['snow_density']

# %%
# get distributions



swe_total_min_q_max = [#data_clipped_fram['snow_depth'].min(dim=['x','y']).resample(time='MS').mean(),
                       data_clipped_fram['snow_depth'].quantile(0.25,dim=['x','y']).resample({time_var:'MS'}).mean(), 
                        data_clipped_fram['snow_depth'].quantile(0.5,dim=['x','y']).resample({time_var:'MS'}).mean(),

                       data_clipped_fram['snow_depth'].quantile(0.75,dim=['x','y']).resample({time_var:'MS'}).mean()]
                       # data_clipped_fram['snow_depth'].max(dim=['x','y']).resample(time='MS').mean()]
                       

# %%

for i,d in enumerate(swe_total_min_q_max):
    d.plot(label=labels[i])
plt.legend()
plt.title('SWE min/max and quartiles')
plt.show()


# %%
def fwh_from_swe(swe, snow_salinity):
    sal_ref = 34.8
    dens_seawater = 1023.9
    return swe*(1-snow_salinity/sal_ref)*(1/dens_seawater)


# %%
sal_height_estimate_different_swe = [fwh_from_swe(x, snow_salinity2) for x in swe_total_min_q_max]
sal_height_estimate_different_swe = [x.to_dataframe() for x in sal_height_estimate_different_swe]

# %%
# sal_height_estimate_different_swe

# %%
fwf_estimate_different_swe = [fram_ice_flux_data['IAF'].mul(x['snow_depth'],axis=0) for x in sal_height_estimate_different_swe]

# %%
# fwf_estimate_different_swe[0]

# %%
# fwf_estimate_different_swe # why did this do that?

# %%
# these are scaled incorrectly??


for i,d in enumerate(fwf_estimate_different_swe):
    d.plot(label=labels[i])
plt.legend()
plt.title('Estimated snow freshwater fluxes using different swe from distribution')
plt.show()


for i,d in enumerate(fwf_estimate_different_depths):
    d.plot(label=labels[i])
plt.legend()
plt.title('Estimated snow freshwater fluxes using different depths from distribution')
plt.show()

# %%
# units got borked somewhere, but anyway I have some options for sensitivity now. 

# %%
# calculate difference relative to median in terms of magnitude
(100*(fwf_estimate_different_swe[2]-fwf_estimate_different_swe[1])/fwf_estimate_different_swe[1]).plot()
# percent difference here up to 250% for some values for smlg

# %%
(100*(fwf_estimate_different_swe[2]-fwf_estimate_different_swe[1])/fwf_estimate_different_swe[1]).mean()

# %%
(100*(fwf_estimate_different_swe[1]-fwf_estimate_different_swe[0])/fwf_estimate_different_swe[1]).plot()


# %%
(100*(fwf_estimate_different_swe[1]-fwf_estimate_different_swe[0])/fwf_estimate_different_swe[1]).mean()

# %%
IQR = fwf_estimate_different_swe[2]-fwf_estimate_different_swe[0]

# %%
IQR.plot()

# %%
(100*(IQR/2)/fwf_estimate_different_swe[1]).plot()

# %%
(100*(IQR/2)/fwf_estimate_different_swe[1]).mean()

# %%
iqr_frac_percent = (100*(IQR/2)/fwf_estimate_different_swe[1])

# %%
# exclude summer months; months = 10 to 4

month_list = [1,2,3,4,10,11,12]



# %%
IQR[IQR.index.month.isin(month_list)].plot()

# %%
iqr_frac_percent[iqr_frac_percent.index.month.isin(month_list)].plot()

# %%
iqr_frac_percent[iqr_frac_percent.index.month.isin(month_list)].mean()

# %%
# differences between products


fwf_snow_ice = pd.read_csv('calculated_fluxes/fram_davis_monthly_snow_ice_fwf.csv',index_col=0,header=[0,1])
fwf_snow_ice.index = pd.to_datetime(fwf_snow_ice.index)

# %%
fwf_snow_ice

# %%
nesosim_smlg_diff = (-100*(fwf_snow_ice['Fram']['SnowModel-LG']-fwf_snow_ice['Fram']['NESOSIM'])/fwf_snow_ice['Fram']['SnowModel-LG'])

# %%
nesosim_smlg_diff.plot()
iqr_frac_percent.plot()


# %%
nesosim_smlg_diff.mean()

# %%
iqr_frac_percent.mean()

# %%
# (IQR[IQR.index.month.isin(month_list)]/2).plot()
# np.abs(-fwf_snow_ice['Fram']['SnowModel-LG']+fwf_snow_ice['Fram']['NESOSIM']).plot()
# np.abs(-fwf_snow_ice['Fram']['SnowModel-LG']+fwf_snow_ice['Fram']['5VLSTM']).plot()
# np.abs(-fwf_snow_ice['Fram']['SnowModel-LG']+fwf_snow_ice['Fram']['NaRRS']).plot()
# np.abs(-fwf_snow_ice['Fram']['SnowModel-LG']+fwf_snow_ice['Fram']['LEGOS']).plot()

# %%
# that's the magnitude as opposed to the percent difference. I think percent difference would matter there?

# %%
