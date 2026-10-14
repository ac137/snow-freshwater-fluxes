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

# %% editable=true slideshow={"slide_type": ""}
######################################################################################
## freshwater_fluxes.ipynb
## script for generating manuscript plots of snow and ice freshwater fluxes 
## and calculating associated aggregate statistics
## uses output from calc_freshwater_fluxes_from_clipped_data.ipynb 
## and aggregate-snow-flux-over-gates.ipynb
## developed by Alex Cabaj (alex.cabaj@ec.gc.ca)
##
## .py file generated using jupytext
##
## Snow datasets used (article reference + doi and dataset doi):
## NESOSIM: NASA Eulerian Snow On Sea Ice Model, v1.1 (Petty et al., 2018; https://doi.org/10.5194/gmd-11-4577-2018)
##        zenodo.org/records/18273881
## SnowModel-LG: Lagrangian Snow Distributions for Sea Ice Applications, version 1 (Liston et al., 2020; https://doi.org/10.1029/2019JC015913)
##       https://nsidc.org/data/nsidc-0758/versions/1
## 5VLSTM: 25 km resolution snow depth on Arctic sea ice dataset (2012-2020) (Li et al., 2022; https://doi.org/10.1016/j.rse.2021.112840)
##        data.tpdc.ac.cn/zh-hans/data/8b4b5f67-ee01-45ee-96a6-db8f712a2e0c
## LEGOS: Multi-frequency altimetry snow depth product over Arctic sea ice (Carret et al., 2025; https://doi.org/10.1038/s41597-024-04343-4)
##        https://doi.org/10.5281/zenodo.14000765
## NaRRS: Nadir Radiometer and Radar Synergy for Arctic Snow Depth and Sea Ice Thickness Retrieval (Nelson et al., 2026;  https://doi.org/10.1029/2025GL121068)
##        https://doi.org/10.5281/zenodo.17942492
#########################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import itertools

import seaborn as sb
import scipy.stats as st
import fwf_helper_functions as ff

# %%
# line colours; from paul tol's colour schemes https://sronpersonalpages.nl/~pault/

line_colours = ['#66CCEE','#4477aa','#228833','#AA3377','#CCBB44','#BBBBBB','#EE6677']
months_ordered = [10,11,12,1,2,3,4]
mon_names = ['Oct','Nov','Dec','Jan','Feb','Mar','Apr']

# if True, saves calculated data as csv files in calculated_fluxes subdirectory
SAVE_OUTPUT_DATA = False

# %%
# load and preprocess ice flux data

fram_ice_flux_data = ff.load_and_preprocess_ice_flux_data('fram-strait-ice-fluxes-v2.csv')
davis_ice_flux_data = ff.load_and_preprocess_ice_flux_data('davis-strait-ice-fluxes-v2.csv')

# %%
# sea ice density (varies based on sea ice age)
sea_ice_density_data = pd.read_csv('monthly_sea_ice_density_from_age.csv',header=[0,1],index_col=0)
sea_ice_density_data.index = pd.to_datetime(sea_ice_density_data.index)

sea_ice_salinity_data = pd.read_csv('monthly_sea_ice_salinity_from_age.csv',header=[0,1],index_col=0)
sea_ice_salinity_data.index = pd.to_datetime(sea_ice_salinity_data.index)

salinity_fram = sea_ice_salinity_data['Fram']['sea_ice_salinity'].loc['2016-10':]
salinity_davis = sea_ice_salinity_data['Davis']['sea_ice_salinity'].loc['2016-10':'2022-04']

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

# %%
# for using fixed-density (prescribed single value) and zero-salinity configurations
# (not shown in manuscript; for sensitivity testing)
USE_FIXED_DENS = False
USE_ZERO_SAL = False

snow_freshwater_height_dict = ff.load_data_into_dict(gate_name_list, data_name_list, file_dataset_list, date_index_name, fixed_dens=USE_FIXED_DENS, zero_sal=USE_ZERO_SAL)
if USE_FIXED_DENS:
    print('USING FIXED DENSITY 293')
if USE_ZERO_SAL:
    print('USING ZERO SALINITY ASSUMPTION FOR ALL DATA')

# %%
# calculate freshwater volume fluxes by gate


davis_freshwater_volume_flux = ff.calculate_ice_freshwater_flux(davis_ice_flux_data['Volume Flux'],
                                                                dens_ice = sea_ice_density_data['Davis']['sea_ice_density'].loc['2016-10':'2022-04'],
                                                               sal_ice = salinity_davis).to_frame(ice_fw_name).loc[:final_date]

davis_freshwater_volume_flux.loc['2017-11','NaRRS'] = np.nan # zero out due to lack of sample points

fram_freshwater_volume_flux = ff.calculate_ice_freshwater_flux(fram_ice_flux_data['VF'],
                                                               dens_ice = sea_ice_density_data['Fram']['sea_ice_density'].loc['2016-10':],
                                                              sal_ice = salinity_fram).to_frame(ice_fw_name).loc[:final_date]

ff.calc_snow_volume_flux(davis_freshwater_volume_flux,snow_freshwater_height_dict['Davis'],davis_ice_flux_data['Area Flux'])
ff.calc_snow_volume_flux(fram_freshwater_volume_flux,snow_freshwater_height_dict['Fram'],fram_ice_flux_data['IAF'])

davis_freshwater_volume_flux *= -1 # convert because otherwise the data is oriented wrong

# %%
davis_freshwater_volume_flux.loc['2017-11','NaRRS'] = np.nan # insufficient samples; zero this out
davis_freshwater_volume_flux = davis_freshwater_volume_flux[fram_freshwater_volume_flux.columns]

# %%
# create a dataframe for just volume and area fluxes

ice_area_volume_fluxes = pd.concat({'Fram': fram_ice_flux_data[['IAF','VF']].rename(columns={'IAF':'Ice area flux','VF':'Ice volume flux'}),
                                    'Davis': davis_ice_flux_data[['Area Flux','Volume Flux']].rename(columns={'Area Flux':'Ice area flux','Volume Flux':'Ice volume flux'})*-1}, 
                                   names=['Gate','Data'],axis=1).loc[:final_date]


# create dataframe with all freshwater fluxes aggregated (also adds area flux)
fw_fluxes_all = ff.combine_fluxes_from_gates(fram_freshwater_volume_flux, davis_freshwater_volume_flux, fram_ice_flux_data, davis_ice_flux_data).loc[:final_date]


# %%
# uncertainties

def calc_fram_davis_uncert():
    # making this a function to keep the namespace clean

    davis_flux_uncert_df1 = davis_freshwater_volume_flux[data_name_list].copy()
    fram_flux_uncert_df1 = fram_freshwater_volume_flux[data_name_list].copy()
    
    davis_flux_uncert_df2 = davis_freshwater_volume_flux[data_name_list].copy()
    fram_flux_uncert_df2 = fram_freshwater_volume_flux[data_name_list].copy()
    # calculate uncertainty volume fluxes from spread of snow fw height values
    # spatial uncertainties
    ff.calc_snow_volume_flux(davis_flux_uncert_df1,snow_freshwater_height_dict['Davis'],davis_ice_flux_data['Area Flux'],'spatial_std')
    ff.calc_snow_volume_flux(fram_flux_uncert_df1,snow_freshwater_height_dict['Fram'],fram_ice_flux_data['IAF'],'spatial_std')
    # temporal uncertainties
    ff.calc_snow_volume_flux(davis_flux_uncert_df2,snow_freshwater_height_dict['Davis'],davis_ice_flux_data['Area Flux'],'temporal_std')
    ff.calc_snow_volume_flux(fram_flux_uncert_df2,snow_freshwater_height_dict['Fram'],fram_ice_flux_data['IAF'],'temporal_std')

    # add uncertainties in quadrature to roughly account for spatial and temporal uncertainties
    fram_flux_uncert_df = np.sqrt(fram_flux_uncert_df1**2 + fram_flux_uncert_df2**2)
    davis_flux_uncert_df = np.sqrt(davis_flux_uncert_df1**2 + davis_flux_uncert_df2**2)
    return fram_flux_uncert_df, davis_flux_uncert_df
    
fram_flux_uncert_df, davis_flux_uncert_df = calc_fram_davis_uncert()

# %%
# add these up and do an annual/seasonal aggregation; calculations, climatologies, annual totals

ice_clim_all, ice_clim_std_all, ice_deseasonalized_all = ff.clim_and_deseasonalize(ice_area_volume_fluxes)


# winter flux totals

winter_flux_totals, winter_flux_totals_sigma, winter_fluxes = ff.calc_winter_flux_data(fw_fluxes_all)

winter_ice_flux_totals, winter_ice_flux_totals_sigma, winter_ice_fluxes = ff.calc_winter_flux_data(ice_area_volume_fluxes)

# %%
# load snow depths for gates (in metres)
# snow depths calculated from aggregate-snow-over-flux-gates.ipynb

gate_snow_depths = {}
gate_snow_depth_err = {}

for i,f in enumerate(file_dataset_list):
    dc = data_name_list[i]
    # snow data calculated using aggregate-snow-over-flux-gates.ipynb
    gate_snow_depths[dc] = pd.read_hdf('snow_depth_swe_gate_mean_monthly_{}.h5'.format(f),key='snow_depth')
    gate_snow_depths[dc].index.name='Date'
    gate_snow_depth_err[dc] = pd.read_hdf('snow_depth_swe_gate_mean_monthly_{}.h5'.format(f),key='snow_depth_std')
    gate_snow_depth_err[dc].index.name='Date'


gate_snow_depths = pd.concat(gate_snow_depths, axis=1).swaplevel(axis=1)[list(itertools.product(gate_name_list,data_name_list))].loc[:final_date]

gate_snow_depth_std = pd.concat(gate_snow_depth_err, axis=1).swaplevel(axis=1)[list(itertools.product(gate_name_list,data_name_list))].loc[:final_date]


# %%
gate_snow_depths.loc['2017-11',('Davis','NaRRS')] = np.nan # zero out due to relative lack of data


# %%
# winter flux annual (winter-season) percentages and associated error
fram_annual_flux_percent = ff.calc_percent(winter_flux_totals['Fram'][data_name_list],winter_flux_totals.loc[:,('Fram',ice_fw_name)])
davis_annual_flux_percent = ff.calc_percent(winter_flux_totals['Davis'][data_name_list],winter_flux_totals.loc[:,('Davis',ice_fw_name)])

fram_annual_flux_percent_sigma, davis_annual_flux_percent_sigma = ff.percent_errprop_gate(winter_flux_totals, winter_flux_totals_sigma, 
                                                                                       gate_name_list, data_name_list, ice_fw_name)



# %%
# sea ice volume flux uncertainties derived from SIT uncert
davis_ice_flux_uncert = (davis_ice_flux_data['Volume Flux']*davis_ice_flux_data['SIT_sigma']/davis_ice_flux_data['SIT']).abs()
fram_ice_flux_uncert = (fram_ice_flux_data['VF']*fram_ice_flux_data['SIT-sigma']/fram_ice_flux_data['SIT']).abs()

# sea ice fwf uncert
davis_ice_fwf_uncert = ff.calculate_ice_freshwater_flux(davis_ice_flux_uncert)
fram_ice_fwf_uncert = ff.calculate_ice_freshwater_flux(fram_ice_flux_uncert)

# %%
# trend calculations (not used in manuscript since no snow fwf trends present)

# trend_df_monthly_freshwater = ff.trend_df_for_all_columns_in_df(fw_fluxes_all)

# winter_ice_flux_trends = ff.trend_df_for_all_columns_in_df(winter_ice_flux_totals)

# winter_fw_trends_all = ff.trend_df_for_all_columns_in_df(winter_flux_totals)

multiproduct_snow_fwf = fw_fluxes_all.loc[:,(slice(None),data_name_list)].groupby(level=0,axis=1).mean()#.mean(axis=1)

# trend in monthly snow multi-product freshwater flux
# trend_df_monthly_snowfwf = ff.trend_df_for_all_columns_in_df(multiproduct_snow_fwf)


# %%
winter_flux_totals#.loc['2016']

# %%
# multiproduct mean time series
davis_snow_fw_multiproduct_mean = davis_freshwater_volume_flux.loc[:,data_name_list].mean(axis=1)
fram_snow_fw_multiproduct_mean = fram_freshwater_volume_flux.loc[:,data_name_list].mean(axis=1)

davis_snow_fw_multiproduct_std = davis_freshwater_volume_flux.loc[:,data_name_list].std(axis=1)
fram_snow_fw_multiproduct_std = fram_freshwater_volume_flux.loc[:,data_name_list].std(axis=1)


# %%
# helper functions for plotting

my_palette = {data_name_list[i]:line_colours[i] for i in range(len(data_name_list))}

# boxplots with different plot for each product
def snowfwf_boxplot_from_fluxes_separate_products(flux_df, gate, ax):
    winter_fluxes_fram = flux_df.loc[:,(gate,data_name_list)].copy()
    winter_fluxes_fram.columns = winter_fluxes_fram.columns.get_level_values(1)
    winter_fluxes_fram['Month'] = winter_fluxes_fram.index.month
    vals_for_plot = winter_fluxes_fram.melt(id_vars=['Month'],value_vars=data_name_list, var_name='Product', value_name='Snow freshwater flux')
    
    sb.boxplot(
        data=vals_for_plot,
        x='Month',
        y='Snow freshwater flux',
        hue='Product',ax=ax,
        palette=my_palette,
        order=months_ordered
    )
    return vals_for_plot

# boxplot with all products consolidated together
def snowfwf_boxplot_from_fluxes(flux_df, gate, ax):
    winter_fluxes_fram = flux_df.loc[:,(gate,data_name_list)].copy()
    winter_fluxes_fram.columns = winter_fluxes_fram.columns.get_level_values(1)
    winter_fluxes_fram['Month'] = winter_fluxes_fram.index.month
    vals_for_plot = winter_fluxes_fram.melt(id_vars=['Month'],value_vars=data_name_list, var_name='Product', value_name='Snow freshwater flux')
    
    sb.boxplot(
        data=vals_for_plot,
        x='Month',
        y='Snow freshwater flux',ax=ax,
        color=line_colours[-2],
        order=months_ordered
    )
    sb.stripplot(
        data=vals_for_plot,
        x='Month',
        y='Snow freshwater flux',ax=ax,
        color='#6C6C6C',
        order=months_ordered,
        jitter=False,
        alpha=0.4
    )
    return vals_for_plot

# helper function for adding axis letter labels to 4-panel plots
def add_ax_letter_labels(ax, text_xpos=0.05, text_ypos = 0.92, text_fontsize=11,right_col_offset=0.03):

    ax[0,0].text(text_xpos, text_ypos, 'a)',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax[0,0].transAxes, fontsize=text_fontsize)
    ax[0,1].text(text_xpos+right_col_offset, text_ypos, 'b)',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax[0,1].transAxes, fontsize=text_fontsize)
    ax[1,0].text(text_xpos, text_ypos, 'c)',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax[1,0].transAxes, fontsize=text_fontsize)
    ax[1,1].text(text_xpos+right_col_offset, text_ypos, 'd)',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax[1,1].transAxes, fontsize=text_fontsize)

def add_asterisk_where_zero(ax_current):
    '''add asterisks/bullets/other marker for patches that have zero value on bar plot'''
    for p in ax_current.patches:
        # print(p.get_height())
        if p.get_height()==0.0:
            # print(p)
            ax_current.annotate('•', (p.get_x(), p.get_height() * 1.005),color=p.get_facecolor(),weight='bold')


# %%
mean_fram_snow = gate_snow_depths['Fram'].mean(axis=1)
std_fram_snow = gate_snow_depths['Fram'].std(axis=1)
mean_davis_snow = gate_snow_depths['Davis'].mean(axis=1)
std_davis_snow = gate_snow_depths['Davis'].std(axis=1)

# %%
# figure 1 b/c: snow thickness over gates time series

# figure 1 b/c: snow thickness over gates time series
alpha_shading = 0.5
fig,ax = plt.subplots(2,1,figsize=(9,5),dpi=300,sharex=True) # for paper
# fig,ax = plt.subplots(2,1,figsize=(8,8),dpi=300,sharex=True) # for presentation

mean_fram_snow.plot(color=line_colours[5],ax=ax[0])


gate_snow_depths['Fram'].plot(ax=ax[0],style='.-',color=line_colours,zorder=4)


ax[0].fill_between(gate_snow_depths['Fram'].index, mean_fram_snow - std_fram_snow, mean_fram_snow + std_fram_snow,
                  color = line_colours[5],alpha=alpha_shading,zorder=-1)

mean_davis_snow.plot(color=line_colours[5],ax=ax[1])


gate_snow_depths['Davis'].plot(ax=ax[1],style='.-',color=line_colours)

ax[1].fill_between(gate_snow_depths['Davis'].index, mean_davis_snow - std_davis_snow, mean_davis_snow + std_davis_snow,
                  color = line_colours[5],alpha=alpha_shading,zorder=-1)


ax[0].set_ylabel('Fram gate-averaged\nsnow depth (m)')
ax[1].set_ylabel('Davis gate-averaged\nsnow depth (m)')
text_xpos = 0.05
text_ypos = 0.85
text_fontsize=11
ax[0].text(text_xpos, text_ypos, 'b)',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax[0].transAxes, fontsize=text_fontsize)
ax[1].text(text_xpos, text_ypos, 'c)',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax[1].transAxes, fontsize=text_fontsize)

ax[1].get_legend().remove()

ax[0].legend(ncol=3,loc=(0.3,0.75))#,loc='upper center')
# ax[0].legend(ncol=2) # for presentation


x0,x1 = ax[0].get_xlim()
ax[0].set_xlim(x0-1,x1+1)
plt.tight_layout()
plt.show()

# %%
################## figure 2: snow freshwater fluxes and monthly (median) clim



fig, ax=plt.subplots(2,2,figsize=(8,6),dpi=300,sharex='col',gridspec_kw={'width_ratios': [2, 1]})

color_right = 'black'

start_date = '2016-09'
end_date = '2024-05'
line_style = '.-'

ax4 = ax[0,0]  
ax4.set_ylabel('Fram Strait monthly snow\nfreshwater volume flux (km$^3$)', color=color_right)

# fram freshwater snow volume flux monthly
plot_df = fram_freshwater_volume_flux.loc[start_date:end_date,data_name_list]
plot_df.plot(style=line_style,ax=ax4,color=line_colours)
# ff.fill_between_err_df(plot_df, fram_flux_uncert_df.loc[start_date:end_date],ax4,color=line_colours)

ax[0,0].legend(loc=(0.11,0.65),ncol=2)
lim0, lim1 = ax4.get_ylim()
upper_lim_2 = 100

# davis freshwater snow volume flux monthly
ax2=ax[1,0]
ax2.set_ylabel('Davis Strait monthly snow\nfreshwater volume flux (km$^3$)', color=color_right)  # we already handled the x-label with ax1

plot_df = davis_freshwater_volume_flux.loc[start_date:end_date,data_name_list]
plot_df.plot(style=line_style,ax=ax2,color=line_colours,legend=None)
# ff.fill_between_err_df(plot_df, davis_flux_uncert_df.loc[start_date:end_date],ax2,color=line_colours)


# adjust snow volume flux axes for davis
lim0, lim1 = ax2.get_ylim()
# # scale y limits so there's more space on top but so they line up at the bottom still
upper_lim_1 = 20

fram_data = snowfwf_boxplot_from_fluxes(winter_fluxes, 'Fram', ax[0,1])

davis_data = snowfwf_boxplot_from_fluxes(winter_fluxes, 'Davis', ax[1,1])
mon_names = ['Oct','Nov','Dec','Jan','Feb','Mar','Apr']

ax[0,1].set_ylabel('Fram monthly snow freshwater\nflux seasonal cycle (km$^3$)')

ax[1,1].set_ylabel('Davis monthly snow freshwater\nflux seasonal cycle (km$^3$)')

add_ax_letter_labels(ax, text_xpos=0.05, text_ypos = 0.92, text_fontsize=11,right_col_offset=0.05)

ax[1,1].set_xticklabels(mon_names)
x0,x1 = ax[0,0].get_xlim()
ax[0,0].set_xlim(x0-1,x1+1)
plt.tight_layout()
plt.show()

# %%
####### snow freshwater flux only, for presentations

fig, ax=plt.subplots(2,1,figsize=(10,8),dpi=300,sharex='col')#,gridspec_kw={'width_ratios': [2, 1]})


color_right = 'black'


start_date = '2016-09'
end_date = '2024-05'
line_style = '.-'

# fram freshwater ice volume flux monthly

ax4 = ax[0]  # instantiate a second Axes that shares the same x-axis
ax4.set_ylabel('Fram snow freshwater volume flux (km$^3$)', color=color_right)  

# fram freshwater snow volume flux monthly
plot_df = fram_freshwater_volume_flux.loc[start_date:end_date,data_name_list]
plot_df.plot(style=line_style,ax=ax4,color=line_colours)
ff.fill_between_err_df(plot_df, fram_flux_uncert_df.loc[start_date:end_date],ax4,color=line_colours)

ax[0].legend(loc=(0.1,0.65))
# adjust snow volume flux axes for fram
lim0, lim1 = ax4.get_ylim()
upper_lim_2 = 100

# davis freshwater snow volume flux monthly
ax2=ax[1]
ax2.set_ylabel('Davis snow freshwater volume flux (km$^3$)', color=color_right)  # we already handled the x-label with ax1

plot_df = davis_freshwater_volume_flux.loc[start_date:end_date,data_name_list]
plot_df.plot(style=line_style,ax=ax2,color=line_colours,legend=None)
ff.fill_between_err_df(plot_df, davis_flux_uncert_df.loc[start_date:end_date],ax2,color=line_colours)

plt.tight_layout()
plt.show()

# %%
############## figure 3: winter season snow fluxes
add_multiyear_totals_to_fig=False

line_alpha=0.6

fig,ax=plt.subplots(2,2,figsize=(11,6.5),dpi=300,sharex='col')

if add_multiyear_totals_to_fig:


    plot_df = pd.concat([winter_flux_totals['Fram'].loc[:'2023',data_name_list], pd.DataFrame([winter_flux_totals['Fram'][data_name_list].mean()],index=[2022])])


    err_df = pd.concat([winter_flux_totals['Fram'].loc[:'2023',data_name_list], pd.DataFrame([winter_flux_totals['Fram'][data_name_list].std()],index=[2022])])
    err_df.loc[2016:2021] = 0

else:
    plot_df = winter_flux_totals.loc[:'2023',('Fram',data_name_list)]
    err_df = None
    

plot_df.plot.bar(color=line_colours,ax=ax[0,0],yerr=err_df)



# winter_flux_totals.loc[:'2023',('Fram',data_name_list)].plot.bar(color=line_colours,ax=ax[0,0])

########### code for adding multiyear totals as horizontal lines
# for i,r in enumerate(winter_flux_totals['Fram'][data_name_list].mean()):

#     ax[0,0].axhline(r,color=line_colours[i],alpha=line_alpha,linestyle='-',zorder=-1,lw=2)#,lw=4)

# ax[0,0].axhline(winter_flux_totals['Fram'][data_name_list].mean()['SnowModel-LG'],color=line_colours[1],alpha=1,dashes=[4, 4],zorder=0,lw=2)
########################


# winter_flux_totals.loc[:'2023',('Davis',data_name_list)].plot.bar(color=line_colours,ax=ax[1,0])

if add_multiyear_totals_to_fig:


    plot_df = pd.concat([winter_flux_totals['Davis'].loc[:'2023',data_name_list], pd.DataFrame([winter_flux_totals['Davis'][data_name_list].mean()],index=[2022])])
    
    err_df = pd.concat([winter_flux_totals['Davis'].loc[:'2023',data_name_list], pd.DataFrame([winter_flux_totals['Davis'][data_name_list].std()],index=[2022])])
    err_df.loc[2016:2021] = 0

else:
    plot_df = winter_flux_totals['Davis'].loc[:'2023',data_name_list]
    err_df = None
plot_df.plot.bar(color=line_colours,ax=ax[1,0],yerr=err_df)




########### code for adding multiyear totals as horizontal lines

# for i,r in enumerate(winter_flux_totals['Davis'][data_name_list].mean()):
#     # print(r)
#     ax[1,0].axhline(r,color=line_colours[i],zorder=-1,alpha=line_alpha,lw=2)

# ax[1,0].axhline(winter_flux_totals['Davis'][data_name_list].mean()['SnowModel-LG'],color=line_colours[1],alpha=1,dashes=[4, 4],zorder=0,lw=2)
#####################################

fram_annual_flux_percent.loc[:'2023'].plot.bar(color=line_colours,ax=ax[0,1])

davis_annual_flux_percent.loc[:'2023'].plot.bar(color=line_colours,ax=ax[1,1])


ax[1,0].set_xlabel('Season')
ax[1,1].set_xlabel('Season')

add_asterisk_where_zero(ax[0,0])
add_asterisk_where_zero(ax[0,1])
add_asterisk_where_zero(ax[1,0])
add_asterisk_where_zero(ax[1,1])

ax[1,0].legend(data_name_list, loc=(0.1,0.73),ncol=2)
ax[0,0].get_legend().remove()
ax[0,1].get_legend().remove()
ax[1,1].get_legend().remove()


ax[0,0].set_ylim(0,170)
ax[1,0].set_ylim(0,45)


add_ax_letter_labels(ax, text_xpos=0.05, text_ypos = 0.92, text_fontsize=11,right_col_offset=0)

ax[0,0].set_ylabel('Fram Strait winter-season\nsnow freshwater flux (km$^3$)')
ax[1,0].set_ylabel('Davis Strait winter-season\nsnow freshwater flux (km$^3$)')
ax[0,1].set_ylabel('Fram Strait winter-season snow\nfreshwater flux as percent of total (%)')
ax[1,1].set_ylabel('Davis Strait winter-season snow\nfreshwater flux as percent of total (%)')
if add_multiyear_totals_to_fig:

    ax[1,0].set_xticklabels(['2016','2017','2018','2019','2020','2021','Multi-year\naverage'])
    ax[0,0].axvline(5.5,color='#777777',ls='--',lw=1) # separate multi-year mean from rest of data

    ax[1,0].axvline(5.5,color='#777777',ls='--',lw=1)
# t = ax[1,0].get_xticks()
plt.tight_layout()
plt.show()

# %%
study_year_end = '2022-04'
# order of averaging matters: first calculate multiproduct mean and then calculate multiyear average

fram_mean_winter_allproduct = winter_flux_totals.loc[:study_year_end,
                                                    ('Fram', data_name_list)].mean(axis=1).mean()
fram_std_winter_allproduct = winter_flux_totals.loc[:study_year_end,
                                                    ('Fram', data_name_list)].mean(axis=1).std()

davis_mean_winter_allproduct = winter_flux_totals.loc[:study_year_end,
                                                    ('Davis', data_name_list)].mean(axis=1).mean()
davis_std_winter_allproduct = winter_flux_totals.loc[:study_year_end,
                                                    ('Davis', data_name_list)].mean(axis=1).std()

# winter averages for annual fwf
fram_ice_fw_mean_winter = winter_flux_totals.loc[:study_year_end,('Fram','Ice freshwater volume flux')].mean()
davis_ice_fw_mean_winter = winter_flux_totals.loc[:study_year_end,('Davis','Ice freshwater volume flux')].mean()

fram_snow_percent_mean_allseason = fram_mean_winter_allproduct*100/(fram_mean_winter_allproduct+fram_ice_fw_mean_winter)
davis_snow_percent_mean_allseason = davis_mean_winter_allproduct*100/(davis_mean_winter_allproduct+davis_ice_fw_mean_winter)

# multiproduct average of winter flux totals
winter_flux_totals[('Fram','Snow freshwater volume flux')] = winter_flux_totals.loc[:,('Fram',data_name_list)].mean(axis=1)
winter_flux_totals[('Davis','Snow freshwater volume flux')] = winter_flux_totals.loc[:,('Davis',data_name_list)].mean(axis=1)

# need to compare fram_mean_winter_allproduct vs winter_flux_totals 'Snow freshwater volume flux' -> these are consistent

# sea ice volume flux uncertainties
davis_ice_flux_uncert = (davis_ice_flux_data['Volume Flux']*davis_ice_flux_data['SIT_sigma']/davis_ice_flux_data['SIT']).abs()
fram_ice_flux_uncert = (fram_ice_flux_data['VF']*fram_ice_flux_data['SIT-sigma']/fram_ice_flux_data['SIT']).abs()

# sea ice fwf uncert
davis_ice_fwf_uncert = ff.calculate_ice_freshwater_flux(davis_ice_flux_uncert)
fram_ice_fwf_uncert = ff.calculate_ice_freshwater_flux(fram_ice_flux_uncert)

# %%
winter_flux_totals

# %%
############## figure 4: snow and ice freshwater fluxes

ADJUST_SNOW_LIM = True
ADJUST_ICE_LIM = True
# if plotting monthly clim; select a single year (ice_clim_all repeats the clim values for the full time series so just subset it)


colours2 = line_colours[5:7][::-1]
colours2 = ['C7','C9']
colours2 = ['C9','C7']

fig, ax=plt.subplots(2,2,figsize=(10,8),dpi=300,sharex='col',gridspec_kw={'width_ratios': [2, 1]})

a_snow=0.7
start_date = '2016-09'
end_date = '2022-04'

ax1 = ax[0,0].twinx()
####### snow multiproduct mean with min/max, Fram
fram_snow_fw_multiproduct_mean.plot(label='Snow multiproduct mean',ax=ax1,color=colours2[1],style='.-',alpha=a_snow)
# ff.fill_between_err_series(fram_snow_fw_multiproduct_mean.loc[start_date:end_date], fram_snow_fw_multiproduct_std.loc[start_date:end_date],
#                     ax1,color=colours2[1])

ff.fill_between_err_series_minmax(fram_snow_fw_multiproduct_mean.loc[start_date:end_date], 
                                   fram_freshwater_volume_flux.loc[:,data_name_list].min(axis=1),
                                   fram_freshwater_volume_flux.loc[:,data_name_list].max(axis=1), ax1,color=colours2[1])
########## fram ice fwf
fram_freshwater_volume_flux.loc[start_date:end_date,ice_fw_name].plot(label='Sea ice freshwater',style='.-',ax=ax[0,0],color=colours2[0])
# volume fwf ice uncert
ff.fill_between_err_series(fram_freshwater_volume_flux.loc[start_date:end_date,ice_fw_name], 
                        fram_ice_fwf_uncert.loc[start_date:end_date],
                        ax=ax[0,0],color=colours2[0])


lines, labels = ax[0,0].get_legend_handles_labels()
lines2, labels2 = ax1.get_legend_handles_labels()
ax1.legend(lines + lines2, ['Sea ice freshwater','Snow multiproduct mean'], loc=(0.1,0.85),ncol=1)
# ax1.legend(lines + lines2, ['Sea ice freshwater','Snow multi-product mean'], loc=(0.1,0.9),ncol=2)

ax1.set_ylabel('Snow freshwater volume flux (km$^3$)', color=colours2[1])

if ADJUST_ICE_LIM:
    lim0, lim1 = ax[0,0].get_ylim()
    # # scale y limits so there's more space on top but so they line up at the bottom still
    upper_lim_1 = 450
    ax[0,0].set_ylim(lim0*upper_lim_1/lim1,upper_lim_1) 
    
if ADJUST_SNOW_LIM:
    lim0, lim1 = ax1.get_ylim()
    # # scale y limits so there's more space on top but so they line up at the bottom still
    upper_lim_1 = 80
    ax1.set_ylim(lim0*upper_lim_1/lim1,upper_lim_1) 

ax2 = ax[1,0].twinx()
ax2.set_ylabel('Snow freshwater volume flux (km$^3$)', color=colours2[1])


######### Davis snow fwf multiproduct mean with uncert
davis_snow_fw_multiproduct_mean.plot(label='Snow multiproduct mean',ax=ax2,color=colours2[1],style='.-',alpha=a_snow)
# ff.fill_between_err_series(davis_snow_fw_multiproduct_mean.loc[start_date:end_date], davis_snow_fw_multiproduct_std.loc[start_date:end_date],
#                     ax2,color=colours2[1])
# fill_between_err_series(davis_snow_fw_multiproduct_mean.loc[start_date:end_date], davis_snow_fw_multiproduct_std.loc[start_date:end_date],
#                     ax2,color=colours2[1])
ff.fill_between_err_series_minmax(davis_snow_fw_multiproduct_mean.loc[start_date:end_date], 
                                   davis_freshwater_volume_flux.loc[:,data_name_list].min(axis=1),
                                   davis_freshwater_volume_flux.loc[:,data_name_list].max(axis=1), ax2,color=colours2[1])


######### Davis ice fwf with uncert
davis_freshwater_volume_flux.loc[start_date:end_date,ice_fw_name].plot(label='Sea ice freshwater',
                                                                       style='.-',ax=ax[1,0],
                                                                       color=colours2[0])
ff.fill_between_err_series(davis_freshwater_volume_flux.loc[start_date:end_date,ice_fw_name], 
                        davis_ice_fwf_uncert.loc[start_date:end_date],
                        ax=ax[1,0],color=colours2[0])

if ADJUST_ICE_LIM:
    lim0, lim1 = ax[1,0].get_ylim()
    # # scale y limits so there's more space on top but so they line up at the bottom still
    upper_lim_1 = 270
    ax[1,0].set_ylim(lim0*upper_lim_1/lim1,upper_lim_1) 

if ADJUST_SNOW_LIM:
    lim0, lim1 = ax2.get_ylim()
    # # scale y limits so there's more space on top but so they line up at the bottom still
    upper_lim_1 = 30
    ax2.set_ylim(lim0*upper_lim_1/lim1,upper_lim_1) 

ax[0,1].set_ylim(0,1300)

ax[1,1].set_ylim(0,750)

winter_flux_totals.loc[:,('Fram',['Ice freshwater volume flux','Snow freshwater volume flux'])].plot.bar(color=colours2,ax=ax[0,1],stacked=True)

winter_flux_totals.loc[:,('Davis',['Ice freshwater volume flux','Snow freshwater volume flux'])].plot.bar(color=colours2,ax=ax[1,1],stacked=True)
ax[1,1].set_xlabel('Season')
ax[1,1].set_ylabel('Davis ice + snow freshwater volume flux (km$^3$)')
ax[0,1].set_ylabel('Fram ice + snow freshwater volume flux (km$^3$)')

add_ax_letter_labels(ax)
x0,x1 = ax[0,0].get_xlim()
ax[0,0].set_xlim(x0-1,x1+1)
# ax[1,0].get_legend().remove()
ax[1,1].get_legend().remove()
ax[0,1].get_legend().remove()
ax[1,0].set_xlabel('Year')
ax[0,0].set_ylabel('Fram Strait ice freshwater volume flux (km$^3$)',color=colours2[0])
ax[1,0].set_ylabel('Davis Strait ice freshwater volume flux (km$^3$)',color=colours2[0])

xtick_label_list = ['2016-\n2017','2017-\n2018','2018-\n2019','2019-\n2020','2020-\n2021','2021-\n2022']
ax[1,1].set_xticklabels(xtick_label_list)
# use max min range here instead for uncert?

for container in ax[0,1].containers:
    ax[0,1].bar_label(container,fmt='%d', label_type='center')

for container in ax[1,1].containers:
    ax[1,1].bar_label(container,fmt='%d', label_type='center')

plt.tight_layout()
plt.show()

# %%
########## save output

if SAVE_OUTPUT_DATA:
    # save monthly gate-average snow depths with uncertainty
    gate_snow_depths.to_csv('calculated_fluxes/fram_davis_monthly_gate_averaged_snow_depth.csv')
    gate_snow_depth_std.to_csv('calculated_fluxes/fram_davis_monthly_gate_averaged_snow_depth_uncert.csv')
    # save monthly snow fwf, monthly sea ice fwf (also includes sea ice area flux) in fram and davis strait
    fw_fluxes_all.to_csv('calculated_fluxes/fram_davis_monthly_snow_ice_fwf.csv')
    # uncertainties in snow fwf for fram and davis strait
    fram_flux_uncert_df.to_csv('calculated_fluxes/fram_monthly_snow_fwf_uncert.csv')
    davis_flux_uncert_df.to_csv('calculated_fluxes/davis_monthly_snow_fwf_uncert.csv')
    # uncertainties in sea ice fwf for fram and davis strait
    fram_ice_fwf_uncert.to_csv('calculated_fluxes/fram_monthly_ice_fwf_uncert.csv')
    davis_ice_fwf_uncert.to_csv('calculated_fluxes/davis_monthly_ice_fwf_uncert.csv')

# %%
############################ Data values, etc

# %%
print('Maximum freshwater flux values with dates of max')
display(fw_fluxes_all.max())
display(fw_fluxes_all.idxmax())


# %%
# full-time-series mean of wintertime monthly snow depth (to give an idea of average depth in each strait)

winter_month_fram_snow = gate_snow_depths['Fram'].index.month.isin([10,11,12,1,2,3,4])
gate_snow_depths['Fram'].loc[winter_month_fram_snow].mean()

# %%
# full-time-series mean of wintertime monthly snow depth (to give an idea of average depth in each strait)

winter_month_davis_snow = gate_snow_depths['Davis'].index.month.isin([10,11,12,1,2,3,4])
gate_snow_depths['Davis'].loc[winter_month_davis_snow].mean()

# %%
print('Maximum snow freshwater flux from all-product average') # max and dates of max
display(multiproduct_snow_fwf.max())

display(multiproduct_snow_fwf.idxmax())


# %%
print('Monthly median winter-season freshwater fluxes')
winter_fluxes.groupby(winter_fluxes.index.month).median()

# %%
print('Months where snow fwf in Davis exceeds Fram for at least one product')
# snow fwf difference between Fram and Davis
df_diff =  fram_freshwater_volume_flux[data_name_list] - davis_freshwater_volume_flux[data_name_list]
df_bool = df_diff<0

bool2 = df_bool.select_dtypes('bool').sum(axis=1).ge(1)

df_diff.loc[bool2].index

# %%
print('Median Fram seasonal cycle snow FWF for all snow datasets')
display(fram_data['Snow freshwater flux'].groupby(fram_data['Month']).median())
print('Median Davis seasonal cycle snow FWF for all snow datasets')
display(davis_data['Snow freshwater flux'].groupby(davis_data['Month']).median())

# %%
print('Interquartile range Fram seasonal cycle snow FWF for all snow datasets')
display((qm:=fram_data['Snow freshwater flux'].groupby(fram_data['Month'])).quantile(0.75)-qm.quantile(0.25))
print('IQR Davis seasonal cycle snow FWF for all snow datasets')
display((qm:=davis_data['Snow freshwater flux'].groupby(davis_data['Month'])).quantile(0.75)-qm.quantile(0.25))

# %%
print('Numerical values of seasonal totals of winter ice area and volume flux')
winter_ice_flux_totals

# %%
print('Numerical values of seasonal totals of winter freshwater volume flux')

winter_flux_totals

# %%
# years of highest winter-total flux
winter_flux_totals.idxmax()

# %%
# check where winter-total is larger in Fram Strait than in Davis Strait
winter_flux_totals['Fram']>winter_flux_totals['Davis']

# %%
# mean of winter flux totals
winter_flux_totals.mean()

# %%
# standard deviation of winter flux totals
winter_flux_totals.std()

# %%
print('winter-season flux percentages in Fram and Davis Strait')
display(fram_annual_flux_percent)
display(davis_annual_flux_percent)

# %%
# True if annual flux percentage (snow/(snow+ice) fwf) is higher in Fram
fram_annual_flux_percent > davis_annual_flux_percent

# %%

# %%

study_year_end = '2022-04'
with pd.option_context('display.float_format', lambda x: f'{x:,.3f}'):
    print('winter flux total multiyear mean')
    df_mean = winter_flux_totals.loc[:study_year_end].mean(axis=0)
    display(df_mean)
    df_std = winter_flux_totals.loc[:study_year_end].std(axis=0)
    print('winter flux total multiyear std')
    display(df_std)
    df_coeff_var = df_std/df_mean
    print('winter flux total multiyear coeff of variation')
    display(df_coeff_var) # coefficient of variation

# %%
# limited to 2018/19 and 2019/20 seasons

study_year_start = '2018'
study_year_end = '2019'
with pd.option_context('display.float_format', lambda x: f'{x:,.3f}'):
    print('winter flux total multiyear mean 18/19-19/20')
    df_mean = winter_flux_totals.loc[study_year_start:study_year_end].mean(axis=0)
    display(df_mean)
    df_std = winter_flux_totals.loc[study_year_start:study_year_end].std(axis=0)
    print('winter flux total multiyear std 18/19-19/20')
    display(df_std)
    # df_coeff_var = df_std/df_mean
    # print('winter flux total multiyear coeff of variation')
    # display(df_coeff_var) # coefficient of variation

# %%
##################### winter totals and percentages over all years
## more decimal places than needed; need to truncate

print('Winter snow freshwater fluxes averaged over all years, \nalso expressed as percent of total snow-plus-ice values')
print('Fram winter all-year all-product snow fwf mean: {:.03f} +- {:.03f}'.format(fram_mean_winter_allproduct,fram_std_winter_allproduct))
print('{:.03f} % of the snow-plus-ice total winter value'.format(fram_snow_percent_mean_allseason))

print('Davis winter all-year all-product snow fwf mean: {:.03f} +- {:.03f}'.format(davis_mean_winter_allproduct,davis_std_winter_allproduct))
print('{:.03f} % of the snow-plus-ice total winter value'.format(davis_snow_percent_mean_allseason))

# %%
# adding ice freshwater flux uncertainties in quadrature
ice_fwf_uncert_fram_seasonal = np.sqrt((fram_ice_fwf_uncert**2).resample('YS-OCT').sum()) # winter totals added in quadrature
ice_fwf_uncert_davis_seasonal = np.sqrt((davis_ice_fwf_uncert**2).resample('YS-OCT').sum()) # winter totals added in quadrature
ice_fwf_uncert_fram_multiyear = np.sqrt((ice_fwf_uncert_fram_seasonal**2).sum())
ice_fwf_uncert_davis_multiyear = np.sqrt((ice_fwf_uncert_davis_seasonal**2).sum())

# %%
ice_fwf_uncert_fram_multiyear, ice_fwf_uncert_davis_multiyear

# %%
print('min/max over different snow products')
print('fram min',winter_flux_totals.loc[:study_year_end,
                                                    ('Fram', data_name_list)].mean(axis=0).min())
print('fram max',winter_flux_totals.loc[:study_year_end,
                                                    ('Fram', data_name_list)].mean(axis=0).max())

print('davis min',winter_flux_totals.loc[:study_year_end,
                                                    ('Davis', data_name_list)].mean(axis=0).min())
print('davis max',winter_flux_totals.loc[:study_year_end,
                                                    ('Davis', data_name_list)].mean(axis=0).max())

# %%
# coefficients of variation: standard deviation / mean of multiyear winter flux totals
# i.e. which time series has more year-to-year differences

df_coeff_var['Fram'].plot(style='.')
df_coeff_var['Davis'].plot(style='.')
plt.title('coefficients of variation for multi-season means of winter total snow fwf')
plt.xticks(rotation=90)
plt.show()

# %%
print('mean multiyear winter ice FWF up to 2021')
winter_flux_totals.loc[:'2021',(slice(None),'Ice freshwater volume flux')].mean()

# %%
print('std multiyear winter ice FWF up to 2021')
winter_flux_totals.loc[:'2021',(slice(None),'Ice freshwater volume flux')].std()

# %%
print('mean multiyear multiproduct winter snow FWF up to 2021')

winter_flux_totals.loc[:'2021',(slice(None),'Snow freshwater volume flux')].mean()

# %%
print('std multiyear multiproduct winter snow FWF up to 2021')

winter_flux_totals.loc[:'2021',(slice(None),'Snow freshwater volume flux')].std()

# %%
# for specific years 2018-2019, 2019-2020

################ percentages for specific year range where all products are available

fram_mean_winter_allproduct_1819 = winter_flux_totals.loc['2018':'2019',
                                                    ('Fram', data_name_list)].mean(axis=1).mean()
fram_std_winter_allproduct_1819 = winter_flux_totals.loc['2018':'2019',
                                                    ('Fram', data_name_list)].mean(axis=1).std()

davis_mean_winter_allproduct_1819 = winter_flux_totals.loc['2018':'2019',
                                                    ('Davis', data_name_list)].mean(axis=1).mean()
davis_std_winter_allproduct_1819 = winter_flux_totals.loc['2018':'2019',
                                                    ('Davis', data_name_list)].mean(axis=1).std()


# sea ice

fram_ice_fw_mean_winter_1819 = winter_flux_totals.loc['2018':'2019',('Fram','Ice freshwater volume flux')].mean()
davis_ice_fw_mean_winter_1819 = winter_flux_totals.loc['2018':'2019',('Davis','Ice freshwater volume flux')].mean()

fram_snow_percent_mean_allseason_1819 = fram_mean_winter_allproduct_1819*100/(fram_mean_winter_allproduct_1819+fram_ice_fw_mean_winter_1819)
davis_snow_percent_mean_allseason_1819 = davis_mean_winter_allproduct_1819*100/(davis_mean_winter_allproduct_1819+davis_ice_fw_mean_winter_1819)

# %%
##################### percentages over 2018-2019

print('Winter snow freshwater fluxes averaged over 2018-2019 and 2019-2020, \nalso expressed as percent of total snow-plus-ice values')
print('Fram winter 2-season all-product snow fwf mean: {:.03f} +- {:.03f}'.format(fram_mean_winter_allproduct_1819,fram_std_winter_allproduct_1819))
print('{:.03f} % of the snow-plus-ice total winter value'.format(fram_snow_percent_mean_allseason_1819))

print('Davis winter 2-season all-product snow fwf mean: {:.03f} +- {:.03f}'.format(davis_mean_winter_allproduct_1819,davis_std_winter_allproduct_1819))
print('{:.03f} % of the snow-plus-ice total winter value'.format(davis_snow_percent_mean_allseason_1819))

# %%
print('min/max over different snow products 2018-2019')
print('fram min',winter_flux_totals.loc['2018':'2019',
                                                    ('Fram', data_name_list)].mean(axis=0).min())
print('fram max',winter_flux_totals.loc['2018':'2019',
                                                    ('Fram', data_name_list)].mean(axis=0).max())

print('davis min',winter_flux_totals.loc['2018':'2019',
                                                    ('Davis', data_name_list)].mean(axis=0).min())
print('davis max',winter_flux_totals.loc['2018':'2019',
                                                    ('Davis', data_name_list)].mean(axis=0).max())

# %%
# are snow fwf more correlated than snow depths (between products) in fram strait? yes if > 0 (excluding diagonal)
fram_freshwater_volume_flux.loc[start_date:end_date,data_name_list].corr() - gate_snow_depths['Fram'].corr()

# %%
# are snow fwf more correlated than snow depths (between products) in davis strait? yes if > 0 (excluding diagonal)

davis_freshwater_volume_flux.loc[start_date:end_date,data_name_list].corr() - gate_snow_depths['Davis'].corr()

# %%
# plot of snow depths minus SMLG (i.e. bias relative to SMLG) to highlight inter-product differences
# in principle could use difference from multi-product mean but this suffices for now

gate_snow_depths['Fram'].sub(gate_snow_depths['Fram']['SnowModel-LG'],axis=0).plot(color=line_colours)
plt.title('Snow depth biases vs SMLG (Fram)')
plt.show()
fram_freshwater_volume_flux.loc[start_date:end_date,data_name_list].sub(fram_freshwater_volume_flux.loc[start_date:end_date,'SnowModel-LG'],axis=0).plot(color=line_colours)
plt.title('Snow fwf biases vs SMLG (Fram)')
plt.show()

# %%

gate_snow_depths['Davis'].sub(gate_snow_depths['Davis']['SnowModel-LG'],axis=0).plot(color=line_colours)
plt.title('Snow depth biases vs SMLG (Davis)')
plt.show()
davis_freshwater_volume_flux.loc[start_date:end_date,data_name_list].sub(davis_freshwater_volume_flux.loc[start_date:end_date,'SnowModel-LG'],axis=0).plot(color=line_colours)
plt.title('Snow fwf biases vs SMLG (Davis)')
plt.show()

# %%
# difference between Fram and Davis snow freshwater flux
(fram_freshwater_volume_flux[data_name_list] - davis_freshwater_volume_flux[data_name_list]).plot()
plt.axhline(0)
plt.title('Fram minus Davis monthly snow freshwater flux difference')
plt.show()

# %%
# snow multi-product mean Fram minus Davis
(fram_freshwater_volume_flux[data_name_list].mean(axis=1) - davis_freshwater_volume_flux[data_name_list].mean(axis=1)).plot()
plt.axhline(0)
plt.title('Fram minus Davis monthly snow freshwater flux difference for multiproduct mean')
plt.show()


# %%
########### want to find out if inter-product consistency is larger between
######### snow depths or snow freshwater fluxes

def compare_cov_and_plot(val1, val2, label1, label2, title):
    std_1 = val1.std(axis=1)
    mean_1 = val1.mean(axis=1)
    std_2 = val2.std(axis=1)
    mean_2 = val2.mean(axis=1)
    (std_1/mean_1).plot(label=label1)
    (std_2/mean_2).plot(label=label2)
    plt.legend()
    plt.title(title)



# %%
# compare inter-product spread
compare_cov_and_plot(gate_snow_depths['Fram'], 
                     fram_freshwater_volume_flux.loc[:,data_name_list], 
                     'Inter-product CoV Fram depth', 
                     'Inter-product CoV Fram snow fwf', 
                     'Inter-product coefficients of variation for snow and snow fwf')

plt.show()

# %%
compare_cov_and_plot(gate_snow_depths['Davis'], 
                     davis_freshwater_volume_flux.loc[:,data_name_list], 
                     'Inter-product CoV Davis depth', 
                     'Inter-product CoV Davis snow fwf', 
                     'Inter-product coefficients of variation for snow and snow fwf')

plt.show()


# %%
def heatmap_plot(df, loc_name,data_var):
    corr = df[loc_name].corr()
    # mask = np.tril(np.ones_like(corr, dtype=bool),k=-1)
    # sb.heatmap(corr, mask=mask,annot=True,cmap='vlag',vmin=-1,vmax=1)
    sb.heatmap(corr,annot=True,cmap='vlag',vmin=-1,vmax=1)

    plt.xlabel('')
    plt.ylabel('')
    plt.title('{} Strait {} correlations'.format(loc_name, data_var))
    plt.show()

def heatmap_plot_2(df, loc_name,data_var):
    corr = df.corr()
    # mask = np.tril(np.ones_like(corr, dtype=bool),k=-1)
    # sb.heatmap(corr, mask=mask,annot=True,cmap='vlag',vmin=-1,vmax=1)
    sb.heatmap(corr,annot=True,cmap='vlag',vmin=-1,vmax=1)

    plt.xlabel('')
    plt.ylabel('')
    plt.title('{} Strait {} correlations'.format(loc_name, data_var))
    plt.show()


# %%
gate_snow_depths['Davis'].corr().mean()

# %%
# inter-product correlations for snow depth and snow fwf

heatmap_plot(gate_snow_depths,'Fram','snow depth temporal')
heatmap_plot_2(fram_freshwater_volume_flux.loc[start_date:end_date,data_name_list],'Fram','snow fwf temporal')


heatmap_plot(gate_snow_depths,'Davis','snow depth temporal')
heatmap_plot_2(davis_freshwater_volume_flux.loc[start_date:end_date,data_name_list],'Davis','snow fwf temporal')

# %%
############## figure : sea ice area and volume fluxes (NOT FRESHWATER, just pure fluxes)


plot_clim=False
# if plotting monthly clim; select a single year (ice_clim_all repeats the clim values for the full time series so just subset it)

ice_clim_for_plot = ice_clim_all.loc['2017-01':'2017-12']#.plot()
ice_clim_for_plot.index = ice_clim_for_plot.index.month

ice_clim_for_plot_sigma = ice_clim_std_all.loc['2017-01':'2017-12']
ice_clim_for_plot_sigma.index = ice_clim_for_plot_sigma.index.month


colours2 = line_colours[3:]

fig, ax=plt.subplots(2,2,figsize=(10,8),dpi=300,sharex='col',gridspec_kw={'width_ratios': [2, 1]})


start_date = '2016-09'
end_date = '2025-05'
ice_area_volume_fluxes.loc[:,(slice(None),'Ice area flux')].plot(style='.-',ax=ax[0,0],color=colours2)
ax[0,0].legend(['Fram','Davis'],loc='upper center',ncol=2)

ice_area_volume_fluxes.loc[:,(slice(None),'Ice volume flux')].plot(style='.-',ax=ax[1,0],color=colours2)


if plot_clim:
    ice_clim_for_plot.loc[:,(slice(None),'Ice area flux')].plot(yerr=ice_clim_for_plot_sigma.loc[:,(slice(None),'Ice area flux')],
                                                                 style='.-',color=colours2,ax=ax[0,1])
    ice_clim_for_plot.loc[:,(slice(None),'Ice volume flux')].plot(yerr=ice_clim_for_plot_sigma.loc[:,(slice(None),'Ice volume flux')],
                                                                   style='.-',color=colours2,ax=ax[1,1])
    ax[1,1].set_xlabel('Month')
    
else:
    

    winter_ice_flux_totals.loc[:,(slice(None),'Ice area flux')].plot.bar(yerr=winter_ice_flux_totals_sigma.loc[:,(slice(None),'Ice area flux')],
                                                                 style='.-',color=colours2,ax=ax[0,1])

    winter_ice_flux_totals.loc[:,(slice(None),'Ice volume flux')].plot.bar(yerr=winter_ice_flux_totals_sigma.loc[:,(slice(None),'Ice volume flux')],
                                                                   style='.-',color=colours2,ax=ax[1,1])
    ax[1,1].set_xlabel('Year')

text_xpos = 0.05
text_ypos = 0.92
text_fontsize = 11
ax[0,0].text(text_xpos, text_ypos, 'a)',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax[0,0].transAxes, fontsize=text_fontsize)
ax[0,1].text(text_xpos+0.03, text_ypos, 'b)',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax[0,1].transAxes, fontsize=text_fontsize)
ax[1,0].text(text_xpos, text_ypos, 'c)',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax[1,0].transAxes, fontsize=text_fontsize)
ax[1,1].text(text_xpos+0.03, text_ypos, 'd)',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax[1,1].transAxes, fontsize=text_fontsize)

ax[1,0].get_legend().remove()
ax[1,1].get_legend().remove()
ax[0,1].get_legend().remove()
ax[1,0].set_xlabel('Year')
ax[0,0].set_ylabel('Sea ice area flux (km$^2$)')
ax[1,0].set_ylabel('Sea ice volume flux (km$^3$)')
plt.tight_layout()
plt.show()


# %%
fig,ax = plt.subplots(2,1,dpi=300,sharex=True)
fram_ice_flux_data['SIT'].plot(ax=ax[0])
ax[0].set_ylabel('Fram Strait SIT')
# plt.show()A
davis_ice_flux_data['SIT'].plot(ax=ax[1])
ax[1].set_ylabel('Davis Strait SIT')
plt.show()

# %%
# same as above  but with area flux

fig, ax=plt.subplots(2,1,figsize=(10,8),dpi=300,sharex='col')#,gridspec_kw={'width_ratios': [2, 1]})

color_left='C7'

color_left = line_colours[-1]

color_right = 'black'


start_date = '2016-09'
end_date = '2024-05'
line_style = '.-'

# fram freshwater ice volume flux
fram_ice_flux_data.loc[start_date:end_date,'IAF'].plot(style='.-',ax=ax[0],color=color_left)

ax[0].set_ylabel('Fram ice area flux (km$^2$)', color=color_left)
ax[1].set_ylabel('Davis ice area flux (km$^2$)', color=color_left)


# davis freshwater ice area flux
(davis_ice_flux_data.loc[start_date:end_date,'Area Flux']*-1).plot(label='Sea ice area ',style='.-',ax=ax[1],color=color_left)

plt.tight_layout()
plt.show()

# %%

# %%
