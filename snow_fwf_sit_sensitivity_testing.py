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
#################################################
# sensitivity testing for freshwater fluxes assuming sit derived using different snow assumptions
# (simple scaling, does not capture all processes etc)
# clip_scaled_sit_to_gates.ipynb and scale_sit_by_snow.ipynb used to calculate input data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import itertools

import seaborn as sb
import scipy.stats as st
import fwf_helper_functions as ff

# %%
line_colours = ['#66CCEE','#AA3377','#CCBB44','#228833','#BBBBBB','#EE6677','#4477aa']
legend_labels = ['NESOSIM','LEGOS','NaRRS','5VLSTM','default']

# %%
sit_data_dict = {}
# sit scaling factors generated using clip_scaled_sit_to_gates.ipynb and scale_sit_by_snow.ipynb used to calculate input data
# note: need to scale sit manually by scaling_term 
for dataset_name in ['nesosim','legos','s3','amsr2']:
    sit_data_dict[dataset_name] = pd.read_csv('cs2_sit_scaled/gate_average_monthly_sit_scaled_{}.csv'.format(dataset_name),index_col=0,header=[0,1])

# %%
sit_data_all = pd.concat(sit_data_dict,axis=1)

# %%
# disregard 'sit'; use 'scaling_term' and apply to sit directly
sit_data_all.index = pd.to_datetime(sit_data_all.index)
# sit_data_all

# %%
sit_data_all = sit_data_all.loc['2016-10':]

# %%
fram_ice_flux_data = ff.load_and_preprocess_ice_flux_data('fram-strait-ice-fluxes-v2.csv')
davis_ice_flux_data = ff.load_and_preprocess_ice_flux_data('davis-strait-ice-fluxes-v2.csv')

# %%
# fram_ice_flux_data

# %%
sit_data_all[('IAF','Fram','iaf')] = fram_ice_flux_data['IAF']
sit_data_all[('IAF','Davis','iaf')] = davis_ice_flux_data['Area Flux']
sit_data_all[('VF','Fram','vf')] = fram_ice_flux_data['VF']
sit_data_all[('VF','Davis','vf')] = davis_ice_flux_data['Volume Flux']
sit_data_all[('SIT','Fram','sit')] = fram_ice_flux_data['SIT']
sit_data_all[('SIT','Davis','sit')] = davis_ice_flux_data['SIT']

# %%
# sit_data_all.loc[:,(slice(None),'Fram','scaling_term')]

# %%
# sit_data_all.loc[:,(slice(None),'Fram','sit')].mul(sit_data_all[('IAF','Fram','iaf')]/1e3,axis=0).plot()
# sit_data_all[('VF','Fram','vf')].plot(label='VF_original')
# plt.title('Fram ice volume fluxes')
# plt.legend()
# plt.show()

# sit_data_all.loc[:,(slice(None),'Fram','sit')].mul(sit_data_all[('IAF','Fram','iaf')]/1e3,axis=0).sub(sit_data_all[('VF','Fram','vf')],axis=0).plot()
# plt.title('Fram ice volume fluxes relative to default')
# plt.show()

# %%
# sit_data_all

# %%
# (sit_data_all.loc[:,(slice(None),'Davis','sit')].mul(sit_data_all[('IAF','Davis','iaf')]/1e3,axis=0)*-1).plot()
# (sit_data_all[('VF','Davis','vf')]*(-1)).plot(label='VF_original')
# plt.title('Davis ice volume fluxes')

# plt.legend()
# plt.show()

# (sit_data_all.loc[:,(slice(None),'Davis','sit')].mul(sit_data_all[('IAF','Davis','iaf')]/1e3,axis=0)*-1).sub(sit_data_all[('VF','Davis','vf')]*(-1),axis=0).plot()
# plt.title('Davis ice volume fluxes relative to default')
# plt.show()

# %%

# %%
# sit_data_all.loc[:,(slice(None),'Fram','sit')].plot()
# plt.title('Fram SIT')
# plt.show()
# sit_data_all.loc[:,(slice(None),'Fram','sit')].sub(sit_data_all.loc[:,('SIT','Fram','sit')],axis=0).plot()
# plt.title('Fram SIT minus default')
# plt.show()

# %%
# sit_data_all.loc[:,(slice(None),'Davis','sit')].sub(sit_data_all.loc[:,('SIT','Davis','sit')],axis=0).plot()
# plt.title('Davis SIT minus default')
# plt.show()

# %%
# load fram and davis flux data
# scale fram sit using scaling term

sit_data_all.loc[:,(slice(None),'Fram','scaling_term')].add(sit_data_all[('SIT','Fram','sit')],axis=0).plot(color = line_colours,style='.-')
sit_data_all[('SIT','Fram','sit')].plot(color='black',style='.-')
plt.title('Fram scaled SIT using scaling term')
plt.legend(labels = legend_labels)
plt.show()

# %%
sit_data_all.loc[:,(slice(None),'Davis','scaling_term')].add(sit_data_all[('SIT','Davis','sit')],axis=0).plot(color = line_colours,style='.-')
sit_data_all[('SIT','Davis','sit')].plot(color='black',style='.-')
plt.title('Davis scaled SIT using scaling term')
plt.legend(labels = legend_labels)

plt.show()

# %%
fram_volume_fluxes_from_snow = sit_data_all.loc[:,(slice(None),'Fram','scaling_term')].add(sit_data_all[('SIT','Fram','sit')],
                                                                                           axis=0).mul(sit_data_all[('IAF','Fram','iaf')],
                                                                                                       axis=0)/1e3

# %%
davis_volume_fluxes_from_snow = sit_data_all.loc[:,(slice(None),'Davis','scaling_term')].add(sit_data_all[('SIT','Davis','sit')],
                                                                                             axis=0).mul(sit_data_all[('IAF','Davis','iaf')],
                                                                                                         axis=0)*-1/1e3

# %%
fig,ax = plt.subplots(2,1,figsize=(10,8),dpi=300)

fram_volume_fluxes_from_snow.plot(color = line_colours,style='.-',ax=ax[0])
sit_data_all[('VF','Fram','vf')].plot(color='black',style='.-',ax=ax[0])
ax[0].legend(labels=legend_labels)
ax[0].set_title('Fram ice volume fluxes scaled by snow')
davis_volume_fluxes_from_snow.plot(color = line_colours,style='.-',ax=ax[1])
(sit_data_all[('VF','Davis','vf')]*-1).plot(color='black',style='.-',ax=ax[1])
plt.legend(loc='lower right',labels=legend_labels)
ax[1].set_title('Davis ice volume fluxes scaled by snow')
plt.tight_layout()
plt.show()

# %%

# %%
# next get snow fwf from these and look at sensitivity/etc? I guess

fwf_snow_ice = pd.read_csv('calculated_fluxes/fram_davis_monthly_snow_ice_fwf.csv',index_col=0,header=[0,1])
fwf_snow_ice.index = pd.to_datetime(fwf_snow_ice.index)

# %%
fwf_snow_ice

# %%
# calculate sea ice freshwater fluxes
# ice_fw_name = 'default_ice_fwf'
# davis_freshwater_volume_flux = ff.calculate_ice_freshwater_flux(davis_ice_flux_data['Volume Flux']).to_frame(ice_fw_name)#.loc[:final_date]
# fram_freshwater_volume_flux = ff.calculate_ice_freshwater_flux(fram_ice_flux_data['VF']).to_frame(ice_fw_name)#.loc[:final_date]

# %%
davis_ice_fwf_from_snow = ff.calculate_ice_freshwater_flux(davis_volume_fluxes_from_snow)
fram_ice_fwf_from_snow = ff.calculate_ice_freshwater_flux(fram_volume_fluxes_from_snow)

# %%
davis_ice_fwf_from_snow_forcomp = davis_ice_fwf_from_snow.copy()

# %%
davis_ice_fwf_from_snow_forcomp['default'] = fwf_snow_ice[('Davis','Ice freshwater volume flux')]

# %%
# davis_ice_fwf_from_snow_forcomp.plot()
# plt.title('Davis ice freshwater flux scaled by snow')
# plt.show()
# # fwf_snow_ice[('Davis','Ice freshwater volume flux')].plot()
# # plt.show()

# %%

# %%
fram_ice_fwf_from_snow_forcomp = fram_ice_fwf_from_snow.copy()
fram_ice_fwf_from_snow_forcomp['default'] = fwf_snow_ice[('Fram','Ice freshwater volume flux')]

# %%
# fram_ice_fwf_from_snow_forcomp.plot()
# plt.title('Fram ice freshwater flux scaled by snow')

# plt.show()

# %%
fig,ax = plt.subplots(2,1,figsize=(10,8),dpi=300)

fram_ice_fwf_from_snow_forcomp.plot(color = line_colours,style='.-',ax=ax[0])
ax[0].legend(labels=legend_labels)
ax[0].set_title('Fram ice freshwater fluxes scaled by snow')
davis_ice_fwf_from_snow_forcomp.plot(color = line_colours,style='.-',ax=ax[1])
plt.legend(loc='lower right',labels=legend_labels)
ax[1].set_title('Davis ice freshwater fluxes scaled by snow')
plt.tight_layout()
plt.show()

# %%
fram_ice_fwf_from_snow_forcomp.plot()
fwf_snow_ice.loc[:,('Fram',('NESOSIM','LEGOS','NaRRS','5VLSTM'))].plot(ax=plt.gca())
plt.title('Ice vs snow freshwater fluxes')
plt.show()

# %%
# def create_unified_snow_ice_fwf_df():

# %%
# now have ice freshwater volume fluxes and snow freshwater volume fluxes, can calculate ratios based on different snow fluxes
# snow divided by ice
davis_snow_flux_ratio_default = ff.calc_percent(fwf_snow_ice[('Davis','SnowModel-LG')], fwf_snow_ice[('Davis','Ice freshwater volume flux')])
davis_snow_flux_ratio_default[davis_snow_flux_ratio_default>99] = np.nan

# %%
davis_ice_fwf_from_snow.columns = davis_ice_fwf_from_snow.columns.droplevel(1)
davis_ice_fwf_from_snow.columns = davis_ice_fwf_from_snow.columns.droplevel(1)
davis_ice_fwf_from_snow = davis_ice_fwf_from_snow.rename(columns={"nesosim": "NESOSIM", "legos": "LEGOS",'s3':'NaRRS','amsr2':'5VLSTM'})

# %%
davis_snow_fwf_for_comparison = fwf_snow_ice.loc[:,('Davis',('NESOSIM','LEGOS','NaRRS','5VLSTM'))]
davis_snow_fwf_for_comparison.columns = davis_snow_fwf_for_comparison.columns.droplevel(0)

# %%
davis_snow_ice_fwf = pd.concat({'ice':davis_ice_fwf_from_snow,'snow':davis_snow_fwf_for_comparison},axis=1)

# %%
davis_snow_ice_fwf # block out values where default ice fwf is less than 0.000000. or less than 1 or something

# %%
# davis_snow_ice_fwf.plot()

# %%
# fram_snow_ice_fwf.plot()

# %%
davis_snow_flux_ratio = ff.calc_percent(davis_snow_ice_fwf['snow'],davis_snow_ice_fwf['ice'])

# %%
davis_snow_flux_ratio.loc['2018']

# %%
fram_snow_flux_ratio_default = ff.calc_percent(fwf_snow_ice[('Fram','SnowModel-LG')], fwf_snow_ice[('Fram','Ice freshwater volume flux')])
fram_snow_flux_ratio_default[fram_snow_flux_ratio_default>99] = np.nan

# %%
fram_ice_fwf_from_snow.columns = fram_ice_fwf_from_snow.columns.droplevel(1)
fram_ice_fwf_from_snow.columns = fram_ice_fwf_from_snow.columns.droplevel(1)
fram_ice_fwf_from_snow = fram_ice_fwf_from_snow.rename(columns={"nesosim": "NESOSIM", "legos": "LEGOS",'s3':'NaRRS','amsr2':'5VLSTM'})

# %%
fram_snow_fwf_for_comparison = fwf_snow_ice.loc[:,('Fram',('NESOSIM','LEGOS','NaRRS','5VLSTM'))]
fram_snow_fwf_for_comparison.columns = fram_snow_fwf_for_comparison.columns.droplevel(0)

# %%
fram_snow_ice_fwf = pd.concat({'ice':fram_ice_fwf_from_snow,'snow':fram_snow_fwf_for_comparison},axis=1)

# %%
fram_snow_flux_ratio = ff.calc_percent(fram_snow_ice_fwf['snow'],fram_snow_ice_fwf['ice'])

# %%
fram_snow_ice_fwf[('ice','default')] = fwf_snow_ice[('Fram','Ice freshwater volume flux')]
fram_snow_ice_fwf[('snow','default')] = fwf_snow_ice[('Fram','SnowModel-LG')]

# %%
davis_snow_ice_fwf[('ice','default')] = fwf_snow_ice[('Davis','Ice freshwater volume flux')]
davis_snow_ice_fwf[('snow','default')] = fwf_snow_ice[('Davis','SnowModel-LG')]

# %%
davis_snow_flux_ratio.plot() # add a column for default ratio? can do stats later
# davis_snow_flux_ratio_default.plot()

# %%
davis_snow_flux_ratio.loc['2021-10':]

# %%
# davis_snow_flux_ratio[davis_snow_flux_ratio>30] = np.nan

# %%
fram_snow_flux_ratio.plot() # add a column for default ratio? can do stats later
# davis_snow_flux_ratio_default.plot()

# %%
fram_snow_flux_ratio.plot() 
davis_snow_flux_ratio.plot(ax=plt.gca()) 
plt.show()

# %%
# 

# %%
# calculate winter flux totals
start_date = '2016-10'
final_date = '2022-04'

davis_snow_ice_fwf.loc[start_date:final_date]#.index.values

# %%
davis_winter_flux_totals, davis_winter_flux_totals_sigma, davis_winter_fluxes = ff.calc_winter_flux_data(davis_snow_ice_fwf.loc[start_date:final_date])
fram_winter_flux_totals, fram_winter_flux_totals_sigma, fram_winter_fluxes = ff.calc_winter_flux_data(fram_snow_ice_fwf.loc[start_date:final_date])


# %%
fram_winter_flux_percent = ff.calc_percent(fram_winter_flux_totals['snow'],fram_winter_flux_totals['ice'])
davis_winter_flux_percent = ff.calc_percent(davis_winter_flux_totals['snow'],davis_winter_flux_totals['ice'])


# %%
fram_winter_flux_percent.plot.bar()

# %%
davis_winter_flux_percent.plot.bar()


# %%
def add_asterisk_where_zero(ax_current):
    '''add asterisks/bullets/other marker for patches that have zero value'''
    for p in ax_current.patches:
        # print(p.get_height())
        if p.get_height()==0.0:
            # print(p)
            ax_current.annotate('•', (p.get_x(), p.get_height() * 1.005),color=p.get_facecolor(),weight='bold')


# %%
fram_winter_flux_percent_default = pd.read_csv('fram_annual_flux_percent.csv',index_col=0)
davis_winter_flux_percent_default = pd.read_csv('davis_annual_flux_percent.csv',index_col=0)
# davis_annual_flux_percent.to_csv('davis_annual_flux_percent.csv')

# %%
fig,ax = plt.subplots(2,1,figsize=(8,8),dpi=300)

fram_winter_flux_percent.plot.bar(color=line_colours,ax=ax[0])#,yerr=err_df)
fram_winter_flux_percent_default[['NESOSIM','LEGOS','NaRRS','5VLSTM','SnowModel-LG']].plot.bar(edgecolor='black',fill=None,ax=ax[0])
davis_winter_flux_percent.plot.bar(color=line_colours,ax=ax[1])
davis_winter_flux_percent_default[['NESOSIM','LEGOS','NaRRS','5VLSTM','SnowModel-LG']].plot.bar(edgecolor='black',fill=None,ax=ax[1])

add_asterisk_where_zero(ax[0])
add_asterisk_where_zero(ax[1])
ax[1].get_legend().remove()
ax[0].legend(labels=['NESOSIM','LEGOS','NaRRS','5VLSTM','SnowModel-LG'])
# ax[0].set_ylim(0,11)
# ax[1].set_ylim(0,11)
ax[0].set_ylabel('Fram Strait snow freshwater flux \nas percent of total (%)')
ax[1].set_ylabel('Davis Strait snow freshwater flux \nas percent of total (%)')
xtick_label_list = ['2016-\n2017','2017-\n2018','2018-\n2019','2019-\n2020','2020-\n2021','2021-\n2022']
ax[0].set_xticklabels(xtick_label_list)
ax[1].set_xticklabels(xtick_label_list)
plt.suptitle('Snow freshwater fluxes as percent of ice flux for snow-scaled ice freshwater fluxes')
plt.tight_layout()
plt.show()


# %%
fram_winter_flux_percent

# %%
scaling_snow_datasets = ['NESOSIM','LEGOS','NaRRS','5VLSTM']

# %%
fram_winter_flux_percent_default[scaling_snow_datasets] - fram_winter_flux_percent[scaling_snow_datasets]

# %%
davis_winter_flux_percent_default[scaling_snow_datasets] - davis_winter_flux_percent[scaling_snow_datasets]

# %%
davis_winter_flux_percent

# %%

# %%
davis_winter_flux_percent_default

# %%
fram_winter_flux_percent.plot.bar()
davis_winter_flux_percent.plot.bar(ax=plt.gca(),color=['C4','C5','C6','C7','C8'])
plt.gca().get_legend().remove()
plt.show()

# %%
fram_winter_flux_percent

# %%
davis_winter_flux_percent

# %%
winter_mean_allyears_fram = fram_winter_flux_totals.mean()
winter_mean_allyears_davis = davis_winter_flux_totals.mean()

# %%
winter_mean_allyears_fram

# %%
# percent for all years

fram_snow_percent_mean_winter = winter_mean_allyears_fram['snow']*100/(winter_mean_allyears_fram['snow']+winter_mean_allyears_fram['ice'])
davis_snow_percent_mean_winter = winter_mean_allyears_davis['snow']*100/(winter_mean_allyears_davis['snow']+winter_mean_allyears_davis['ice'])

# %%
fram_snow_percent_mean_winter

# %%
davis_snow_percent_mean_winter

# %%
# fram_snow_percent_mean_winter.plot.bar()
# davis_snow_percent_mean_winter.plot.bar(ax=plt.gca(),color='C2')
# plt.title('Fram (blue) and Davis (green) Strait wintertime\nsnow freshwater fluxes from Oct 2016-Apr 2022')
# plt.show()

# %%
fram_snow_percent_mean_winter.mean()

# %%
davis_snow_percent_mean_winter.mean()

# %%
# try manual scaling

fram_ice_flux_data

# %%



snow_depth = pd.read_csv('calculated_fluxes/fram_davis_monthly_gate_averaged_snow_depth.csv',index_col=0,header=[0,1])
snow_depth.index = pd.to_datetime(snow_depth.index)

# %%
# examine snow depths

# %%
# examine snow depth from 

# %%
