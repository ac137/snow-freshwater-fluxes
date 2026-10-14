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
#######################################
# script for comparing snow FWF calculated using 
# prescribed snow density and variable density values
# along with prescribed and variable salinity values
#######################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import itertools

import seaborn as sb
import scipy.stats as st
import fwf_helper_functions as ff

# %%


line_colours = ['#66CCEE','#4477aa','#228833','#AA3377','#CCBB44','#BBBBBB','#EE6677']
months_ordered = [10,11,12,1,2,3,4]
mon_names = ['Oct','Nov','Dec','Jan','Feb','Mar','Apr']

# %%
# load and preprocess ice flux data



fram_ice_flux_data = ff.load_and_preprocess_ice_flux_data('fram-strait-ice-fluxes-v2.csv')
davis_ice_flux_data = ff.load_and_preprocess_ice_flux_data('davis-strait-ice-fluxes-v2.csv')

# including LEGOS, s3, and smlg m2
file_dataset_list = ['nesosim','smlg-m2','amsr2','legos_sam', 's3']
data_name_list = ['NESOSIM','SnowModel-LG','5VLSTM','LEGOS', 'NaRRS']
gate_name_list = ['Fram','Davis']
date_index_name = ['day','time','date','time','time']
ice_fw_name = 'Ice freshwater volume flux'

# adding modified warren 99; note mw99 has some incomplete coverage here
# file_dataset_list = ['nesosim','smlg','amsr2','legos_sam', 's3','mw99']
# data_name_list = ['NESOSIM','SnowModel-LG','5VLSTM','LEGOS', 'NaRRS','mW99']
# gate_name_list = ['Fram','Davis']
# date_index_name = ['day','time','date','time','time','time']
# ice_fw_name = 'Ice freshwater volume flux'


final_date = '2022-04'


# %%
def load_and_consolidate_fwf(fixed_dens=False,zero_sal=False):
    # function for loading; relies on existence of various variables (would need refactoring before using as helper function)
    snow_freshwater_height_dict = ff.load_data_into_dict(gate_name_list, data_name_list, file_dataset_list, date_index_name, fixed_dens=fixed_dens,zero_sal=zero_sal)
    davis_freshwater_volume_flux = ff.calculate_ice_freshwater_flux(davis_ice_flux_data['Volume Flux']).to_frame(ice_fw_name).loc[:final_date]
    fram_freshwater_volume_flux = ff.calculate_ice_freshwater_flux(fram_ice_flux_data['VF']).to_frame(ice_fw_name).loc[:final_date]
    
    ff.calc_snow_volume_flux(davis_freshwater_volume_flux,snow_freshwater_height_dict['Davis'],davis_ice_flux_data['Area Flux'])
    ff.calc_snow_volume_flux(fram_freshwater_volume_flux,snow_freshwater_height_dict['Fram'],fram_ice_flux_data['IAF'])
    
    davis_freshwater_volume_flux *= -1 # convert because otherwise the data is oriented wrong

    fw_fluxes_all = ff.combine_fluxes_from_gates(fram_freshwater_volume_flux, davis_freshwater_volume_flux, fram_ice_flux_data, davis_ice_flux_data).loc[:final_date]
    return fw_fluxes_all


# %%
fwf_variable_density = load_and_consolidate_fwf(fixed_dens=False) # variable snow density, variable snow salinity (this is the configuration used in the main text)
fwf_fixed_density = load_and_consolidate_fwf(fixed_dens=True) # fixed snow density
fwf_0sal = load_and_consolidate_fwf(fixed_dens=False,zero_sal=True) # zero snow salinity


# %%

# %%
def limit_months_to_study_period(df):
    df[df.index.month.isin([5,6,7,8,9])] = np.nan
    return df
    # return df[df.index.month.isin([10,11,12,1,2,3,4])]


# %%
fwf_variable_density = limit_months_to_study_period(fwf_variable_density)
fwf_fixed_density = limit_months_to_study_period(fwf_fixed_density)
fwf_0sal = limit_months_to_study_period(fwf_0sal)


# %%
def heatmap_plot(df, loc_name, data_var):
    sb.heatmap(df.loc[:,(loc_name,data_name_list)].corr(),annot=True,cmap='vlag',vmin=-1,vmax=1)
    plt.xlabel('')
    plt.ylabel('')
    plt.title('{} Strait {} correlations'.format(loc_name, data_var))
    plt.show()

def heatmap_diff_plot(df1, df2, loc_name, data_var,vmax=0.1):
    sb.heatmap((df1.loc[:,(loc_name,data_name_list)].corr() - df2.loc[:,(loc_name,data_name_list)].corr()),annot=True,cmap='vlag',vmin=-vmax,vmax=vmax)
    plt.xlabel('')
    plt.ylabel('')
    plt.title('{} Strait {} correlations'.format(loc_name, data_var))
    plt.show()

# %%
# heatmap_plot(fwf_variable_density, 'Fram', 'variable density')
# heatmap_plot(fwf_fixed_density, 'Fram', 'fixed density')

# heatmap_plot(fwf_variable_density, 'Davis', 'variable density')
# heatmap_plot(fwf_fixed_density, 'Davis', 'fixed density')



# %%

# heatmap_diff_plot(fwf_variable_density,fwf_fixed_density, 'Fram', 'variable - fixed density',0.05)
# heatmap_diff_plot(fwf_variable_density,fwf_fixed_density, 'Davis', 'variable - fixed density',0.05)

# %%
# heatmap_plot(fwf_variable_density, 'Fram', 'variable salinity')
# heatmap_plot(fwf_0sal, 'Fram', 'zero salinity')

# heatmap_plot(fwf_variable_density, 'Davis', 'variable salinity')
# heatmap_plot(fwf_0sal, 'Davis', 'zero salinity')

# heatmap_diff_plot(fwf_variable_density,fwf_0sal, 'Fram', 'full - zero salinity',0.05)
# heatmap_diff_plot(fwf_variable_density,fwf_0sal, 'Davis', 'full - zero salinity',0.05)

# %%
# fram mean bias due to density

(fwf_variable_density - fwf_fixed_density)['Fram'][data_name_list].mean() # mean bias due to density fram

# %%
# fwf_fixed_density['Fram'][data_name_list].mean()

# %%
# fram rms due to density
np.sqrt(((fwf_variable_density - fwf_fixed_density)**2)['Fram'][data_name_list].mean()) # rms

# %%
# fram rms as percent density
100*np.sqrt(((fwf_variable_density - fwf_fixed_density)**2)['Fram'][data_name_list].mean())/fwf_fixed_density['Fram'][data_name_list].mean()# rms as percent*100

# %%
# (fwf_variable_density - fwf_fixed_density)['Fram'][data_name_list].mean()

# %%
# percent mean bias due to density fram
(fwf_variable_density - fwf_fixed_density)['Fram'][data_name_list].mean()/fwf_fixed_density['Fram'][data_name_list].mean()*100 # percent mean bias due to density fram

# %%
# mean bias due to density davis
(fwf_variable_density - fwf_fixed_density)['Davis'][data_name_list].mean() # mean bias due to density davis

# %%
# davis density rms
np.sqrt(((fwf_variable_density - fwf_fixed_density)**2)['Davis'][data_name_list].mean()) # rms

# %%
# rms as percent*100 davis dens
100*np.sqrt(((fwf_variable_density - fwf_fixed_density)**2)['Davis'][data_name_list].mean())/fwf_fixed_density['Davis'][data_name_list].mean()

# %%
# percent mean bias due to density fram

(fwf_variable_density - fwf_fixed_density)['Davis'][data_name_list].mean()/fwf_fixed_density['Davis'][data_name_list].mean()*100 # percent mean bias due to density fram

# %%
# mean bias due to salinity fram
(fwf_variable_density - fwf_0sal)['Fram'][data_name_list].mean() #mean bias due to salinity fram

# %%
# percent bias due to salinity fram
(fwf_variable_density - fwf_0sal)['Fram'][data_name_list].mean()/fwf_0sal['Fram'][data_name_list].mean()*100 # percent bias due to salinity fram

# %%

# %%
# rms  fram sal
np.sqrt(((fwf_variable_density - fwf_0sal)**2)['Fram'][data_name_list].mean())#/fwf_0sal['Fram'][data_name_list].mean()

# %%
# rms as percent*100 fram sal
100*np.sqrt(((fwf_variable_density - fwf_0sal)**2)['Fram'][data_name_list].mean())/fwf_0sal['Fram'][data_name_list].mean()

# %%
# mean bias due to salinity davis
(fwf_variable_density - fwf_0sal)['Davis'][data_name_list].mean()

# %%
# percent bias due to salinity davis
(fwf_variable_density - fwf_0sal)['Davis'][data_name_list].mean()/fwf_0sal['Davis'][data_name_list].mean()*100

# %%
# rms davis sal
np.sqrt(((fwf_variable_density - fwf_0sal)**2)['Davis'][data_name_list].mean())#/fwf_0sal['Davis'][data_name_list].mean()

# %%
# rms davis percent sal
100*np.sqrt(((fwf_variable_density - fwf_0sal)**2)['Davis'][data_name_list].mean())/fwf_0sal['Davis'][data_name_list].mean()

# %%
fwf_variable_density['Fram'][data_name_list].plot()

# %%


# plt.show()
# (fwf_variable_density - fwf_fixed_density)['Davis'][data_name_list].plot(color=line_colours,marker='.')
# plt.title('Davis Strait variable minus fixed density')

# plt.show()

fig,ax = plt.subplots(2,1, dpi=300,sharey=False)

(fwf_variable_density - fwf_fixed_density)['Fram'][data_name_list].plot(color=line_colours,marker='.',ax=ax[0])
# ax[0].set_title('Fram Strait variable minus fixed density')
# (fram_freshwater_volume_flux_dens-fram_freshwater_volume_flux).plot(ax=ax[0])
# # ax[0].legend(labels = ['variable density','fixed density'])
# (davis_freshwater_volume_flux_dens - davis_freshwater_volume_flux).plot(ax=ax[1])
# ax[0].legend().remove()
ax[0].legend(loc='upper left',ncol=3)
(fwf_variable_density - fwf_fixed_density)['Davis'][data_name_list].plot(color=line_colours,marker='.',ax=ax[1])

ax[0].set_title('Fram Strait snow freshwater volume flux difference (variable - fixed density)')
ax[1].set_title('Davis Strait snow freshwater volume flux difference (variable - fixed density)')
# ax[0].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')
# ax[1].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')
ax[1].legend().remove()

plt.tight_layout()
plt.show()

fig,ax = plt.subplots(2,1, dpi=300,sharey=False)

(fwf_variable_density - fwf_0sal)['Fram'][data_name_list].plot(color=line_colours,marker='.',ax=ax[0])
# ax[0].set_title('Fram Strait variable minus fixed density')
# (fram_freshwater_volume_flux_dens-fram_freshwater_volume_flux).plot(ax=ax[0])
# # ax[0].legend(labels = ['variable density','fixed density'])
# (davis_freshwater_volume_flux_dens - davis_freshwater_volume_flux).plot(ax=ax[1])
# ax[0].legend().remove()
ax[0].legend(loc='lower left',ncol=3)
(fwf_variable_density - fwf_0sal)['Davis'][data_name_list].plot(color=line_colours,marker='.',ax=ax[1])

ax[0].set_title('Fram Strait snow freshwater volume flux difference (variable - zero salinity)')
ax[1].set_title('Davis Strait snow freshwater volume flux difference (variable - zero salinity)')
# ax[0].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')
# ax[1].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')
ax[1].legend().remove()

plt.tight_layout()
plt.show()


# (fwf_variable_density - fwf_0sal)['Fram'][data_name_list].plot(color=line_colours,marker='.')
# plt.title('Fram Strait full minus zero salinity')

# plt.show()
# (fwf_variable_density - fwf_0sal)['Davis'][data_name_list].plot(color=line_colours,marker='.')
# plt.title('Davis Strait full minus zero salinity')

# plt.show()

# %%

# %%
((fwf_variable_density - fwf_fixed_density)*100/fwf_fixed_density)['Fram'][data_name_list].plot(color=line_colours)
plt.title('Fram Strait variable minus fixed density % difference')

plt.show()
((fwf_variable_density - fwf_fixed_density)*100/fwf_fixed_density)['Davis'][data_name_list].plot(color=line_colours)
plt.title('Davis Strait variable minus fixed density % difference')

plt.show()


((fwf_variable_density - fwf_0sal)*100/fwf_0sal)['Fram'][data_name_list].plot(color=line_colours)
plt.title('Fram Strait full minus zero salinity % difference')

plt.show()
((fwf_variable_density - fwf_0sal)*100/fwf_0sal)['Davis'][data_name_list].plot(color=line_colours)
plt.title('Davis Strait full minus zero salinity % difference')

plt.show()

# %%

plt.figure(dpi=300)
fwf_variable_density['Fram'][data_name_list].plot(color=line_colours,marker='.',alpha=0.5,ax=plt.gca())
fwf_fixed_density['Fram'][data_name_list].plot(ax=plt.gca(),color=line_colours,style=':',marker='.')
plt.title('Snow FWF in Fram Strait;\n variable (solid) vs fixed (dashed) density')

plt.legend(ncol=2)

plt.show()


# %%
def calc_coeff_var(df):
    return df.std(axis=1)/np.abs(df.mean(axis=1))


# %%
def plot_coeff_var(df1, df2, label1, label2, region_name, value_name):
    v1 = calc_coeff_var(df1)
    v2 = calc_coeff_var(df2)
    plt.figure(dpi=300)
    v1.plot(color=line_colours,marker='.',alpha=0.5,ax=plt.gca(),label=label1)
    v2.plot(ax=plt.gca(),color=line_colours,style=':',marker='.',label=label2)
    plt.title('{} {} (solid) vs {} (dashed) {} inter-product spread'.format(region_name,label1, label2, value_name))
    plt.legend(ncol=2)
    plt.show()
    plt.figure(dpi=300)
    (v1 - v2).plot(color=line_colours,marker='.',alpha=0.5,ax=plt.gca())
    plt.title('{} {} (solid) minus {} (dashed) {} inter-product spread'.format(region_name,label1, label2, value_name))
    plt.axhline(0)
    plt.show()



# %%
plot_coeff_var(fwf_variable_density['Fram'][data_name_list], fwf_fixed_density['Fram'][data_name_list], 'variable','fixed', 'Fram Strait', 'density')


# %%

# %%
plt.figure(dpi=300)

fwf_variable_density['Davis'][data_name_list].plot(ax=plt.gca(),color=line_colours,marker='.',alpha=0.5)
fwf_fixed_density['Davis'][data_name_list].plot(ax=plt.gca(),color=line_colours,style=':',marker='.')
plt.title('Snow FWF in Davis Strait;\n variable (solid) density vs fixed (dashed) density')
plt.legend(ncol=2)
plt.show()

# %%
plot_coeff_var(fwf_variable_density['Davis'][data_name_list], fwf_fixed_density['Davis'][data_name_list], 'variable','fixed', 'Davis Strait', 'density')



# %%
plt.figure(dpi=300)

fwf_variable_density['Fram'][data_name_list].plot(ax=plt.gca(),color=line_colours,marker='.',alpha=0.5)
fwf_0sal['Fram'][data_name_list].plot(ax=plt.gca(),color=line_colours,style=':',marker='.')

plt.title('Snow FWF in Fram Strait; variable salinity (solid) vs zero salinity (dashed)')

plt.legend(ncol=2)
plt.show()

# %%
plot_coeff_var(fwf_variable_density['Fram'][data_name_list], fwf_0sal['Fram'][data_name_list], 'full','zero', 'Fram Strait', 'salinity')


# %%
plt.figure(dpi=300)

fwf_variable_density['Davis'][data_name_list].plot(ax=plt.gca(),color=line_colours,marker='.',alpha=0.5)
fwf_0sal['Davis'][data_name_list].plot(ax=plt.gca(),color=line_colours,style=':',marker='.')
plt.title('Snow FWF in Davis Strait; variable salinity (solid) vs zero salinity (dashed)')

plt.legend(ncol=2)
plt.show()

# %%
plot_coeff_var(fwf_variable_density['Davis'][data_name_list], fwf_0sal['Davis'][data_name_list], 'full','zero', 'Davis Strait', 'salinity')


# %%
# impact on fwf; want to quantify that I guess

# %%

# %%
