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
ice_fw_name = 'Ice freshwater volume flux'
final_date = '2022-04'

# %%


fram_ice_flux_data = ff.load_and_preprocess_ice_flux_data('fram-strait-ice-fluxes-v2.csv')
davis_ice_flux_data = ff.load_and_preprocess_ice_flux_data('davis-strait-ice-fluxes-v2.csv')



# %%
# davis_ice_flux_data

# %%
# def sal_from_sit(sit):
#     # sal_all = sit.copy()
#     sal_all = 14.24 - 19.39*sit
#     sal_all[sit>0.4] = 7.88-1.59*sit[sit>0.4]
#     return sal_all
    # sal_all[sit<=0.4] = 
    # if sit > 0.4:
    #     sal = 7.88-1.59*sit
    # else: # this would also include nan
    #     sal = 14.24 - 19.39*sit

# %%
# salinity_fram = sal_from_sit(fram_ice_flux_data['SIT'])#.plot()
# salinity_davis = sal_from_sit(davis_ice_flux_data['SIT'])

# %%
# sal_from_sit(davis_ice_flux_data['SIT']).plot()

# %%
# fram_ice_flux_data

# %%
# davis_ice_flux_data

# %%
# from calc_ice_density_salinity_gate_avg.ipynb

sea_ice_density_data = pd.read_csv('monthly_sea_ice_density_from_age.csv',header=[0,1],index_col=0)
sea_ice_density_data.index = pd.to_datetime(sea_ice_density_data.index)


# %%
dens_fram = sea_ice_density_data['Fram']['sea_ice_density'].loc['2016-10':]
dens_davis = sea_ice_density_data['Davis']['sea_ice_density'].loc['2016-10':'2022-04']


# %%
sea_ice_salinity_data = pd.read_csv('monthly_sea_ice_salinity_from_age.csv',header=[0,1],index_col=0)
sea_ice_salinity_data.index = pd.to_datetime(sea_ice_salinity_data.index)

sal_fram_1 = sea_ice_salinity_data['Fram']['sea_ice_salinity'].loc['2016-10':]
sal_davis_1 = sea_ice_salinity_data['Davis']['sea_ice_salinity'].loc['2016-10':'2022-04']

# %%
salinity_fram = sal_fram_1
salinity_davis = sal_davis_1

# %%
davis_freshwater_volume_flux_dens = ff.calculate_ice_freshwater_flux(davis_ice_flux_data['Volume Flux'],
                                                                     dens_ice=dens_davis).to_frame(ice_fw_name).loc[:final_date]*-1

fram_freshwater_volume_flux_dens = ff.calculate_ice_freshwater_flux(fram_ice_flux_data['VF'],
                                                                    dens_ice=dens_fram).to_frame(ice_fw_name).loc[:final_date]


# %%
davis_freshwater_volume_flux = ff.calculate_ice_freshwater_flux(davis_ice_flux_data['Volume Flux']).to_frame(ice_fw_name).loc[:final_date]*-1
fram_freshwater_volume_flux = ff.calculate_ice_freshwater_flux(fram_ice_flux_data['VF']).to_frame(ice_fw_name).loc[:final_date]


# %%
davis_freshwater_volume_flux_sal = ff.calculate_ice_freshwater_flux(davis_ice_flux_data['Volume Flux'],
                                                                   sal_ice=salinity_davis).to_frame(ice_fw_name).loc[:final_date]*-1
fram_freshwater_volume_flux_sal = ff.calculate_ice_freshwater_flux(fram_ice_flux_data['VF'],
                                                                    sal_ice=salinity_fram).to_frame(ice_fw_name).loc[:final_date]


# %%
davis_freshwater_volume_flux_sal_dens = ff.calculate_ice_freshwater_flux(davis_ice_flux_data['Volume Flux'],
                                                                    dens_ice=dens_davis,
                                                                   sal_ice=salinity_davis).to_frame(ice_fw_name).loc[:final_date]*-1
fram_freshwater_volume_flux_sal_dens = ff.calculate_ice_freshwater_flux(fram_ice_flux_data['VF'],
                                                                         dens_ice=dens_fram,
                                                                    sal_ice=salinity_fram).to_frame(ice_fw_name).loc[:final_date]


# %%
plt.figure(dpi=300)
sea_ice_density_data.plot(ax=plt.gca())
# plt.legend(labels=['Fram','Davis'])
plt.title('Gate-average sea ice density')
plt.ylabel('Sea ice density (kg m$^{-3}$)')
plt.axhline(901.3,color='C3',label='fixed value')
plt.legend(loc=[0.01,0.7],labels=['Fram','Davis','fixed value'])

plt.show()

# %%
plt.figure(dpi=300)
# sea_ice_density_data.plot(ax=plt.gca())
salinity_fram.plot(ax=plt.gca())
salinity_davis.plot(ax=plt.gca())
# plt.legend(labels=['Fram','Davis'])
plt.title('Gate-average sea ice salinity')
plt.ylabel('Sea ice salinity (PSU)')
plt.axhline(4,color='C3',label='fixed value')
plt.legend(loc=[0.01,0.7],labels=['Fram','Davis','fixed value'])

plt.show()

# %%
fig,ax = plt.subplots(2,1, dpi=300)
(fram_freshwater_volume_flux_dens).plot(label='variable density',ax=ax[0])
(fram_freshwater_volume_flux).plot(ax=ax[0], label='fixed density')
ax[0].legend(labels = ['variable density','fixed density'])
(davis_freshwater_volume_flux_dens).plot(ax=ax[1])
(davis_freshwater_volume_flux).plot(ax=ax[1])
ax[1].legend().remove()

ax[0].set_title('Fram Strait sea ice freshwater volume flux for variable and fixed sea ice density')
ax[1].set_title('Davis Strait sea ice freshwater volume flux for variable and fixed sea ice density')
ax[0].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')
ax[1].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')

plt.tight_layout()
plt.show()

# %%
fig,ax = plt.subplots(2,1, dpi=300)
(fram_freshwater_volume_flux_sal).plot(label='variable salinity',ax=ax[0])
(fram_freshwater_volume_flux).plot(ax=ax[0], label='fixed salinity')
ax[0].legend(labels = ['variable salinity','fixed salinity'])
(davis_freshwater_volume_flux_sal).plot(ax=ax[1])
(davis_freshwater_volume_flux).plot(ax=ax[1])
ax[1].legend().remove()

ax[0].set_title('Fram Strait sea ice freshwater volume flux for variable and fixed sea ice salinity')
ax[1].set_title('Davis Strait sea ice freshwater volume flux for variable and fixed sea ice salinity')
ax[0].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')
ax[1].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')

plt.tight_layout()
plt.show()

# %%
fig,ax = plt.subplots(2,1, dpi=300)
(fram_freshwater_volume_flux_sal_dens).plot(label='variable both',ax=ax[0])
(fram_freshwater_volume_flux).plot(ax=ax[0], label='fixed all')
ax[0].legend(labels = ['variable both','fixed all'])
(davis_freshwater_volume_flux_sal_dens).plot(ax=ax[1])
(davis_freshwater_volume_flux).plot(ax=ax[1])
ax[1].legend().remove()

ax[0].set_title('Fram Strait sea ice freshwater volume flux for variable and fixed sea ice salinity/density')
ax[1].set_title('Davis Strait sea ice freshwater volume flux for variable and fixed sea ice salinity/density')
ax[0].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')
ax[1].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')

plt.tight_layout()
plt.show()


# %%
def calc_rms(x1,x2):
    return np.sqrt(np.mean((x1-x2)**2))

def calc_mb(x1,x2):
    return np.mean(x2-x1)


# %%
# rms difference # fram
calc_rms(fram_freshwater_volume_flux_dens,fram_freshwater_volume_flux)


# %%
100*calc_rms(fram_freshwater_volume_flux_dens,fram_freshwater_volume_flux)/fram_freshwater_volume_flux_dens.mean()

# %%
# davis rms
calc_rms(davis_freshwater_volume_flux_dens,davis_freshwater_volume_flux)


# %%
100*calc_rms(davis_freshwater_volume_flux_dens,davis_freshwater_volume_flux)/davis_freshwater_volume_flux_dens.mean()


# %%
# mean bias fram
calc_mb(fram_freshwater_volume_flux, fram_freshwater_volume_flux_dens)



# %%
# mean bias davis
calc_mb(davis_freshwater_volume_flux,davis_freshwater_volume_flux_dens)


# %%
fig,ax = plt.subplots(2,1, dpi=300,sharey=True)
(fram_freshwater_volume_flux_dens-fram_freshwater_volume_flux).plot(ax=ax[0])
# ax[0].legend(labels = ['variable density','fixed density'])
(davis_freshwater_volume_flux_dens - davis_freshwater_volume_flux).plot(ax=ax[1])
ax[0].legend().remove()

ax[1].legend().remove()

ax[0].set_title('Fram Strait sea ice freshwater volume flux difference (variable - fixed density)')
ax[1].set_title('Davis Strait sea ice freshwater volume flux difference (variable - fixed density)')
ax[0].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')
ax[1].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')

plt.tight_layout()
plt.show()

# %%
fig,ax = plt.subplots(2,1, dpi=300,sharey=True)
(fram_freshwater_volume_flux_sal-fram_freshwater_volume_flux).plot(ax=ax[0])
# ax[0].legend(labels = ['variable density','fixed density'])
(davis_freshwater_volume_flux_sal - davis_freshwater_volume_flux).plot(ax=ax[1])
ax[0].legend().remove()

ax[1].legend().remove()

ax[0].set_title('Fram Strait sea ice freshwater volume flux difference (variable - fixed salinity)')
ax[1].set_title('Davis Strait sea ice freshwater volume flux difference (variable - fixed salinity)')
ax[0].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')
ax[1].set_ylabel('Ice freshwater\nvolume flux (km$^3$)')

plt.tight_layout()
plt.show()

# %%
# calculate all the different rms etc values also

# %%
# rms difference # fram
calc_rms(fram_freshwater_volume_flux_sal,fram_freshwater_volume_flux)


# %%
100*calc_rms(fram_freshwater_volume_flux_sal,fram_freshwater_volume_flux)/fram_freshwater_volume_flux_sal.mean()

# %%
calc_rms(davis_freshwater_volume_flux_sal,davis_freshwater_volume_flux)


# %%
100*calc_rms(davis_freshwater_volume_flux_sal,davis_freshwater_volume_flux)/davis_freshwater_volume_flux_sal.mean()

# %%
# mean bias fram
calc_mb(fram_freshwater_volume_flux, fram_freshwater_volume_flux_sal)



# %%
calc_mb(davis_freshwater_volume_flux, davis_freshwater_volume_flux_sal)


# %%

# %%

# %%
