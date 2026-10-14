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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import itertools

import seaborn as sb
import scipy.stats as st
import fwf_helper_functions as ff

# %%
# datasets to load, data names for plots, index names
# smlg driven by merra-2
file_dataset_list = ['nesosim','smlg-m2','amsr2','legos_sam', 's3']
data_name_list = ['NESOSIM','SnowModel-LG','5VLSTM','LEGOS', 'NaRRS']
gate_name_list = ['Fram','Davis']
date_index_name = ['day','time','date','time','time']
ice_fw_name = 'Ice freshwater volume flux'

# last date to examine
final_date = '2022-04'

fram_ice_flux_data = ff.load_and_preprocess_ice_flux_data('fram-strait-ice-fluxes-v2.csv')
davis_ice_flux_data = ff.load_and_preprocess_ice_flux_data('davis-strait-ice-fluxes-v2.csv')

sea_ice_density_data = pd.read_csv('monthly_sea_ice_density_from_age.csv',header=[0,1],index_col=0)
sea_ice_density_data.index = pd.to_datetime(sea_ice_density_data.index)

sea_ice_salinity_data = pd.read_csv('monthly_sea_ice_salinity_from_age.csv',header=[0,1],index_col=0)
sea_ice_salinity_data.index = pd.to_datetime(sea_ice_salinity_data.index)

salinity_fram = sea_ice_salinity_data['Fram']['sea_ice_salinity'].loc['2016-10':]
salinity_davis = sea_ice_salinity_data['Davis']['sea_ice_salinity'].loc['2016-10':'2022-04']

density_fram = sea_ice_density_data['Fram']['sea_ice_density'].loc['2016-10':]
density_davis = sea_ice_density_data['Davis']['sea_ice_density'].loc['2016-10':'2022-04']

USE_FIXED_DENS = False
USE_ZERO_SAL = False

snow_freshwater_height_dict = ff.load_data_into_dict(gate_name_list, data_name_list, file_dataset_list, date_index_name, fixed_dens=USE_FIXED_DENS, zero_sal=USE_ZERO_SAL)
if USE_FIXED_DENS:
    print('USING FIXED DENSITY 293')
if USE_ZERO_SAL:
    print('USING ZERO SALINITY ASSUMPTION FOR ALL DATA')


# %%
# get snow freshwater height and sea ice freshwater height
# have snow freshwater height

# sea ice freshwater height

def calc_salinity_height(height, salinity, density):
    '''calculate freshwater water-equivalent height by scaling by salinity and density
    uses prescribed reference salinity (sal_ref) and assumed seawater density (dens_seawater)'''
    sal_ref = 34.8
    dens_seawater = 1023.9
    return height*(1-salinity/sal_ref)*(density/dens_seawater)


# %%
# have sea ice salinity, sea ice height is in fram_ice_flux_data/davis_ice_flux_data
# sea ice density is in sea_ice_density_data

ice_salinity_height_fram = calc_salinity_height(fram_ice_flux_data['SIT'], salinity_fram, density_fram).loc[:final_date]
ice_salinity_height_davis = calc_salinity_height(davis_ice_flux_data['SIT'], salinity_davis, density_davis)

# %%
ice_salinity_height_fram.plot()

# %%
ice_salinity_height_davis.plot()

# %%
# snow_freshwater_height_dict['Fram']['SnowModel-LG']['freshwater_height']#
# fram_fw_snow_test = snow_freshwater_height_dict['Fram']['SnowModel-LG']['freshwater_height']
# davis_fw_snow_test = snow_freshwater_height_dict['Davis']['SnowModel-LG']['freshwater_height']

# %%
snow_fwh_fram_df = pd.concat(snow_freshwater_height_dict['Fram'],axis=1).drop(columns=['spatial_std','temporal_std','day','date','time'],level=1)
snow_fwh_davis_df = pd.concat(snow_freshwater_height_dict['Davis'],axis=1).drop(columns=['spatial_std','temporal_std','day','date','time'],level=1)

snow_fwh_fram_df.columns = snow_fwh_fram_df.columns.droplevel(1)
snow_fwh_davis_df.columns = snow_fwh_davis_df.columns.droplevel(1)

# %%
snow_fwh_df = pd.concat({'Fram':snow_fwh_fram_df,'Davis':snow_fwh_davis_df},axis=1)
snow_fwh_df

# %%
# fram_ratio = fram_fw_snow_test*100/(ice_salinity_height_fram+fram_fw_snow_test)
# fram_ratio.plot()

# %%
# davis_ratio = davis_fw_snow_test/(ice_salinity_height_davis+davis_fw_snow_test)
# davis_ratio[davis_ratio>0.99] = np.nan
# davis_ratio = davis_ratio*100

# davis_ratio.plot() # filter values >0.99

# %%
# fram_ratio.plot()
# davis_ratio.plot() # this is just for smlg for now but just for reference

# %%
snow_fwh_clim, snow_fwh_std, snow_fwh_deseason = ff.clim_and_deseasonalize(snow_fwh_df)

# %%
snow_fwh_deseason['Fram'].plot()

# %%
# sea ice fwh
ice_fwh = pd.concat({'Fram':ice_salinity_height_fram, 'Davis':ice_salinity_height_davis},axis=1)

# %%
ice_fwh_clim, ice_fwh_std, ice_fwh_deseason = ff.clim_and_deseasonalize(ice_fwh)

# %%
ice_fwh_deseason.plot()

# %%
snow_fwh_deseason['Fram'].corrwith(ice_fwh_deseason['Fram']) # deseasonalized snow fwh vs deseasonalized sea ice fwh

# %%
snow_fwh_deseason['Davis'].corrwith(ice_fwh_deseason['Davis'])

# %%
snow_fwh_deseason_withice = snow_fwh_deseason.copy()
snow_fwh_deseason_withice[('Fram','sea ice')] = ice_fwh_deseason['Fram']
snow_fwh_deseason_withice[('Davis','sea ice')] = ice_fwh_deseason['Davis']

# %%
snow_fwh_deseason_withice.plot.scatter(x=('Fram','NESOSIM'),y=('Fram','sea ice'))

# %%
snow_fwh_deseason_withice.plot.scatter(x=('Davis','NESOSIM'),y=('Davis','sea ice'))

# %%
fram_snow_ratio = snow_fwh_df['Fram']/snow_fwh_df['Fram'].add(ice_fwh['Fram'],axis=0)

# %%
fram_snow_multi_ratio = snow_fwh_df['Fram'].mean(axis=1) / snow_fwh_df['Fram'].mean(axis=1).add(ice_fwh['Fram'],
                                                                                                axis=0)
davis_snow_multi_ratio = snow_fwh_df['Davis'].mean(axis=1) / snow_fwh_df['Davis'].mean(axis=1).add(ice_fwh['Davis'],
                                                                                                axis=0)
davis_snow_multi_ratio[davis_snow_multi_ratio>0.99] = np.nan

# %%
davis_snow_ratio = snow_fwh_df['Davis']/snow_fwh_df['Davis'].add(ice_fwh['Davis'],axis=0)
davis_snow_ratio[davis_snow_ratio>0.99] = np.nan

# %%
fram_snow_ratio.plot()
plt.show()

# %%
davis_snow_ratio.plot()
plt.show()

# %%
plt.figure(dpi=300)
ax=plt.gca()
fram_snow_ratio[:final_date].plot(ax=ax)
davis_snow_ratio[:final_date].plot(style='--',ax=ax,color=['C0','C1','C2','C3','C4','C5'])
plt.legend(ncol=2)
plt.show()

# %%
plt.figure(dpi=300)
ax=plt.gca()
fram_snow_multi_ratio[:final_date].plot(ax=ax)
davis_snow_multi_ratio[:final_date].plot(style='--',ax=ax,color=['C0','C1','C2','C3','C4','C5'])
plt.legend(ncol=2)
plt.show()

# %%
snow_fwh_df['Fram'].mean(axis=1).corrwith(ice_fwh['Fram'])

# %%
# should we be including winter months only for this?

# %%
snow_fwh_df['Fram']['SnowModel-LG']['2017-04':]

# %%
ice_fwh['2017-04':]

# %%
snow_fwh_df.count()

# %%
ice_fwh.count()

# %%
snow_fwh_df['Fram']['NESOSIM'].autocorr()


# %%
# plot autocorrelation to examine length scale

def calc_autocorr_in_arr(data_col):
    n = data_col.count()
    c_all = []
    n_all = []
    for i in range(1,n+1):
        n_all.append(i)
        c_all.append(data_col.autocorr(i))
    return n_all, c_all


# %%
# snow_fwh_df[c].autocorr(0)

# %%
e_scale = 1/np.exp(1)
for c in snow_fwh_df.columns:
    print(c)
    x,y = calc_autocorr_in_arr(snow_fwh_df[c])
    plt.plot(x,y,'.')
    plt.axhline(e_scale)
    plt.show()
    # print(c,snow_fwh_df[c].autocorr())
    # print(c,snow_fwh_df[c].autocorr(2))
    # print(c,snow_fwh_df[c].autocorr(3))

# %%
x,y = calc_autocorr_in_arr(snow_fwh_df['Fram'].mean(axis=1))
plt.plot(x,y,'.')
plt.axhline(e_scale)
plt.show()

x,y = calc_autocorr_in_arr(snow_fwh_df['Davis'].mean(axis=1))
plt.plot(x,y,'.')
plt.axhline(e_scale)
plt.show()

# %%
snow_fwh_df['Fram'].mean(axis=1).count() # max is 67 since that's count of ice fwh

# %%
snow_fwh_df['Davis'].mean(axis=1).count() # max is 67 since that's length of ice fwh

# %%
# non deseasonalized snow-ice freshwater height correlation
ice_fwh.corrwith(snow_fwh_df['Fram'].mean(axis=1))

# %%
ice_fwh.corrwith(snow_fwh_df['Davis'].mean(axis=1))

# %%
# ice FWH deseasonalized correlated with snow multiproduct mean
# for Fram; davis column kept because pandas was giving errors otherwise, use only 'fram' result
ice_fwh_deseason.corrwith(snow_fwh_deseason['Fram'].mean(axis=1))

# %%
# ice FWH deseasonalized correlated with snow multiproduct mean
# for Davis; fram column kept because pandas was giving errors otherwise, use only 'davis' result
ice_fwh_deseason.corrwith(snow_fwh_deseason['Davis'].mean(axis=1))

# %%
# ice freshwater and snow freshwater correlations in Fram - put these in a supplement
snow_fwh_deseason['Fram'].corrwith(ice_fwh_deseason['Fram']) 

# %%
# ice freshwater and snow freshwater correlations in Davis - put these in a supplement

snow_fwh_deseason['Davis'].corrwith(ice_fwh_deseason['Davis']) # put these in a supplement

# %%
# excluding NESOSIM
ice_fwh_deseason.corrwith(snow_fwh_deseason['Fram'][['SnowModel-LG','5VLSTM','LEGOS','NaRRS']].mean(axis=1))

# %%
snow_fwh_deseason['Fram'].count()

# %%
snow_fwh_deseason['Davis'].count()

# %%
