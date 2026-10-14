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
############################################################################## 
## batch_clip_to_shapefile_intermediate.ipynb
## script for batch clipping data to gate shapefiles
## .py file generated from ipynb using jupytext
## uses output from snow_salinity_calculation.ipynb, regrid_sea_ice_age.ipynb
## shapefiles must have same crs as data
#############################################################################

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

# %%
MASK_OUT_MYI = True
year = 2016

# dataset_name = 'smlg-m2'
# dataset_name = 'legos_sam'
# dataset_name = 'amsr2'
dataset_name = 's3'
# dataset_name = 'nesosim'
# dataset_name = 'mw99' # mw99 is experimental - not used

# %%
def clip_data_in_dict(data_dict, gate_mask_dict):
    # function for clipping data that has been loaded into dictionary data_dict
    # clip to gate masks in gate_mask_dict
    # nested dictionary structured as gate: variable name
    gate_clipped_data = {}
    # iterate through gate names
    for gate_name, model_mask in gate_mask_dict.items():
        # create sub-dictionary for gate name
        print(gate_name)
        gate_clipped_data[gate_name] = {}

        # iterate through model names
        # for model_name, model_mask in model_mask_dict.items():
        #     # create sub-dictionary for model name
        #     gate_clipped_data[gate_name][model_name] = {}
        #     print(model_name)
            
            # iterate through individual model datasets now
            # model mask is from loop above
        for variable_name, model_data in data_dict.items():
            print(variable_name)
            gate_clipped_data[gate_name][variable_name] = model_data.where(model_mask==1)
            # gate_masked_data_list = v2.where(gate_mask==1)
    return gate_clipped_data

# maybe see if the dict manipulation here can be streamlined
def load_and_preprocess_data(data_name):
    data_dict_by_model = {}

    # wrapping data loading in a function so that memory gets freed up afterwards; not all these vars are needed
    year=2016
    MASK_OUT_MYI = True
    if data_name == 'nesosim':

        # load datasets
        # load snow depth/density
        nesosim_da = xr.open_mfdataset('../snow_modelling/nesosim_default_data/*.nc')
        # load salinity
        nesosim_sal = xr.open_dataset('nesosim_snow_salinity_estimate.nc') # taken from snow_salinity_calculation.ipynb
        # load myi fraction
        data_myi_frac = xr.open_dataset('monthly_myi_fraction_nesosim_grid.nc') # taken from regrid_sea_ice_age.ipynb

        # setting x/y grids for nesosim (not necessary for other products generally)
        nesosim_da['xgrid'] = nesosim_da['xgrid'][0] # do this if using mfdataset for nesosim, otherwise do not
        nesosim_da['ygrid'] = nesosim_da['ygrid'][0]
    
        nesosim_sal['xgrid'] = nesosim_da['xgrid'] # adjust x grids and y grids to be consistent just in case
        nesosim_sal['ygrid'] = nesosim_da['ygrid']
    
        # create dict for raw model data
        model_raw_data = {'nesosim_data':nesosim_da, 
                  'nesosim_salinity':nesosim_sal}

        # add crs and dates as needed
        for k,v in model_raw_data.items():

            model_raw_data[k] = sc.add_crs_and_dates_nesosim(v)

        # dictionary for data variable names
        data_var_dict = {
              'nesosim_data':['snow_depth','snow_density'],
              'nesosim_salinity':['snow_salinity']
             }
        # dictionary for names of data
        data_name_dict = data_var_dict


    
    elif data_name == 'smlg':
        smlg_da_snod = xr.open_dataset('../SnowModel_data/SM_snod_ERA5_01Aug1980-31Jul2021_v01.nc')
        smlg_da_sden = xr.open_dataset('../SnowModel_data/SM_sden_ERA5_01Aug1980-31Jul2021_v01.nc')
        smlg_sal = xr.open_dataset('smlg_snow_salinity_estimate.nc')
        data_myi_frac = xr.open_dataset('monthly_myi_fraction_SMLG_grid.nc')

        model_raw_data = {'smlg_depth':smlg_da_snod,
                  'smlg_density':smlg_da_sden,'smlg_salinity':smlg_sal}
        for k,v in model_raw_data.items():
            # add crs to all the datasets (SMLG crs)
            model_raw_data[k] = sc.add_crs_snowmodel(v)
        # names of variables in datasets loaded from files
        data_var_dict = {'smlg_depth':['snod'],'smlg_density':['sden'],
                      'smlg_salinity':['snow_salinity'],
                     }
        # new names for variables to be saved as
        data_name_dict = {'smlg_depth':['snow_depth'],'smlg_density':['snow_density'],
                      'smlg_salinity':['snow_salinity']
                     }
    elif data_name == 'smlg-m2':
        smlg_da_snod = xr.open_dataset('../SnowModel_data/SM_snod_MERRA2_01Aug1980-31Jul2021_v01.nc')
        smlg_da_sden = xr.open_dataset('../SnowModel_data/SM_sden_MERRA2_01Aug1980-31Jul2021_v01.nc')
        smlg_sal = xr.open_dataset('smlg_snow_salinity_estimate_m2.nc')
        data_myi_frac = xr.open_dataset('monthly_myi_fraction_SMLG_grid.nc')

        model_raw_data = {'smlg_depth':smlg_da_snod,
                  'smlg_density':smlg_da_sden,'smlg_salinity':smlg_sal}
        for k,v in model_raw_data.items():
            # add crs to all the datasets (SMLG crs)
            model_raw_data[k] = sc.add_crs_snowmodel(v)
        # names of variables in datasets loaded from files
        data_var_dict = {'smlg_depth':['snod'],'smlg_density':['sden'],
                      'smlg_salinity':['snow_salinity'],
                     }
        # new names for variables to be saved as
        data_name_dict = {'smlg_depth':['snow_depth'],'smlg_density':['snow_density'],
                      'smlg_salinity':['snow_salinity']
                     }

    elif data_name =='amsr2':
        # pmw taken from model_snow_obs_comparisons/consolidate_pmw_tif_files.ipynb
        amsr2_li_smlg_grid = xr.open_dataset('../snow_modelling/model_snow_obs_comparisons/pmw_amsr2_li_smlg_grid.nc') # li et al data on smlg grid
        smlg_da_sden = xr.open_dataset('../SnowModel_data/SM_sden_MERRA2_01Aug1980-31Jul2021_v01.nc') # using smlg density
        amsr2_sal = xr.open_dataset('amsr2_li_snow_salinity_estimate.nc') # amsr2 salinity on smlg grid
        # using myi fraction on smlg grid
        data_myi_frac = xr.open_dataset('monthly_myi_fraction_SMLG_grid.nc')
        # select bands for amsr2 data
        amsr2_li_smlg_grid = amsr2_li_smlg_grid.sel(band=1)
        amsr2_sal = amsr2_sal.sel(band=1)

        model_raw_data = {
                  'smlg_density':smlg_da_sden,
                  'amsr2_smlg_depth':amsr2_li_smlg_grid, 'amsr2_smlg_salinity':amsr2_sal}
        for k,v in model_raw_data.items():
            # add crs to all the datasets (SMLG crs)
            model_raw_data[k] = sc.add_crs_snowmodel(v)

        # dictionary of data variable names vs raw model data names
        data_var_dict = {'smlg_density':['sden'],
                     'amsr2_smlg_depth':['snow_depth'],
                     'amsr2_smlg_salinity':['snow_salinity']
                     }
        # dictionary of data variable names to variables used for output
        data_name_dict = {'smlg_density':['snow_density'],
                 'amsr2_smlg_depth':['snow_depth'],
                 'amsr2_smlg_salinity':['snow_salinity']
                 }
    elif data_name == 'legos_sam':
        legos_data = xr.open_mfdataset('../other_data/SD_LaKu_LEGOS_NH_Carret_and_Fleury/*.nc')
        legos_salinity = xr.open_dataset('legos_sam_snow_salinity_estimate.nc') 
        data_myi_frac = xr.open_dataset('monthly_myi_fraction_legos_grid.nc')
        # rename x/y coords; specific to legos
        # rename_dict = {'xc':'y','yc':'x'}
        rename_dict = {'xc':'x','yc':'y'}
        legos_data = legos_data.rename(rename_dict)
        legos_salinity = legos_salinity.rename(rename_dict)
        data_myi_frac = data_myi_frac.rename(rename_dict)

        # y_new = legos_data['x'][::-1]
        # x_new = legos_data['y'][::-1]
        x_tmp = legos_data['x']
        y_tmp = legos_data['y']
        legos_data['x']= y_tmp.values
        legos_data['y']=x_tmp.values
        legos_salinity['x']= y_tmp.values
        legos_salinity['y']=x_tmp.values
        data_myi_frac['x']= y_tmp.values
        data_myi_frac['y']=x_tmp.values




        model_raw_data = {'legos_data': legos_data,
                          'legos_salinity': legos_salinity}
        for k,v in model_raw_data.items():

            model_raw_data[k] = sc.add_crs_legos(v)
        # names of variables to load from xarrays
        data_var_dict = {
              'legos_data':['snow_depth_laku_sam','snow_density'],
              'legos_salinity':['snow_salinity']
             }
        # names to save data variables with
        data_name_dict = {'legos_data':['snow_depth','snow_density'],
              'legos_salinity':['snow_salinity']
                         }
    elif data_name == 's3':
        s3_data = xr.open_mfdataset('../snow-freshwater-fluxes/s3_aggregated/s3_aggregated_snow_depth_20*.nc')
        s3_salinity = xr.open_dataset('s3_snow_salinity_estimate.nc') 
        data_myi_frac = xr.open_dataset('monthly_myi_fraction_s3_grid.nc')
        s3_density = xr.open_mfdataset('../snow-freshwater-fluxes/s3_aggregated/s3_aggregated_snow_density_20*.nc')
        # s3_density[:] = 293.33
        # s3_density = s3_density.to_dataset(name='snow_density')

        model_raw_data = {
                  's3_density':s3_density,
                  's3_depth':s3_data, 's3_salinity':s3_salinity}
        for k,v in model_raw_data.items():

            model_raw_data[k] = sc.add_crs_legos(v) # has same crs as legos (epsg 6931)
        data_var_dict = {'s3_depth':['snow_depth'],'s3_density':['snow_density'],
                      's3_salinity':['snow_salinity']
                     }
        # new names for variables to be saved as
        data_name_dict = data_var_dict
    elif data_name == 'mw99':
        # note: would need to manually generate mw99 rather than use the is2 product; data has gaps
        mw99_data = xr.open_mfdataset('../IS2SITMOGR4_v4/IS2SITMOGR4_01_*.nc').sortby('time')
        mw99_density = xr.open_dataset('mw99_snow_density_is2_grid.nc') # calculated from test_is2_sitmogr4.ipynb to be interpolated
        mw99_salinity = xr.open_dataset('mw99_snow_salinity_estimate.nc') 
        data_myi_frac = xr.open_dataset('monthly_myi_fraction_is2_grid.nc')

        # depth and density are in the same ds so putting these together
        model_raw_data = {'mw99_data':mw99_data, 'mw99_density':mw99_density,
                          'mw99_salinity':mw99_salinity}
        for k,v in model_raw_data.items():

            model_raw_data[k] = sc.add_crs_is2(v) # epsg 3411
        # variables to load from each dataset
        data_var_dict = {'mw99_data':['snow_depth_mw99_int'], 
                         'mw99_density':['snow_density'],
                      'mw99_salinity':['snow_salinity']
                     }
        # new names for variables to be saved as
        data_name_dict = {'mw99_data':['snow_depth'],
                          'mw99_density':['snow_density'],
                      'mw99_salinity':['snow_salinity']
                     }


    # this should be dataset-agnostic
    data_preformatted_dict = {}
    
    for k,v in model_raw_data.items():
        # go through raw data
        # list of variables to iterate through
        variable_list = data_var_dict[k]
        for i, var_name in enumerate(variable_list):
            # go through the variable list and slice from the xarray
            if data_name == 'mw99':
                data_processed = v[var_name].loc['{}-09-01'.format(year):'2024-12'] # truncate long time series here
            else:
                data_processed = v[var_name].loc['{}-09-01'.format(year):]
            # key for the new dictionary (new dataset name)
            key_new = data_name_dict[k][i]
            # now assign to new data. don't need to keep initializing vars
            # but doing so for clarity
            data_preformatted_dict[key_new] = data_processed.to_dataset(name=key_new)
        # data_dict_by_model = {data_name:data_preformatted_dict}

        # add myi fraction
        data_preformatted_dict['myi_fraction']=data_myi_frac

    if data_name == 'amsr2':
        data_preformatted_dict['snow_depth']/= 100 # scale amsr2 snow depth
        # subset data to end in dec 2020 for amsr2
        data_preformatted_dict['myi_fraction']=data_myi_frac.sel(time=slice(None,'2020-12'))
        # select time-coincident densities for amsr2 (don't always have the right dates available)
        data_preformatted_dict['snow_density']=data_preformatted_dict['snow_density'].sel(time=data_preformatted_dict['snow_depth']['date'].dt.date)
        

    return data_preformatted_dict

# %%
preprocessed_data_dict = load_and_preprocess_data(dataset_name)

# %%
buffer_size = 50

# if buffer_size==30:
#     shapefile_names_nesosim = ['shapefiles/FramGate30kmBufferPoly.shp', 'shapefiles/davis_gate_buffer.shp']
#     shapefile_names_smlg = ['shapefiles/fram-buffer-epsg-3408.shp','shapefiles/davis-buffer-epsg-3408.shp']
if buffer_size==50:
    # different shapefiles for different crs
    shapefile_names_nesosim = ['shapefiles/fram-buffer-50km-3413.shp','shapefiles/davis-buffer-50km-3413.shp']
    shapefile_names_smlg = ['shapefiles/fram-buffer-50km-3408.shp','shapefiles/davis-buffer-50km-3408.shp']
    shapefile_names_legos = ['shapefiles/fram-buffer-50km-6931.shp','shapefiles/davis-buffer-50km-6931.shp']
    shapefile_names_is2 = ['shapefiles/fram-buffer-50km-3411.shp','shapefiles/davis-buffer-50km-3411.shp']
# elif buffer_size==80:
#     shapefile_names_nesosim = ['shapefiles/fram-buffer-larger-3413.shp','shapefiles/davis-buffer-larger-3413.shp']
#     shapefile_names_smlg = ['shapefiles/fram-buffer-larger-3408.shp','shapefiles/davis-buffer-larger-3408.shp']
else:
    print('wrong size')
    



# %%
# single file clipping to set up binary grid masks
# since all input products are on common grids; much faster than clipping individually
if dataset_name == 'nesosim':
    shapefiles_nesosim = sc.open_shapefiles_from_list(shapefile_names_nesosim)
    nesosim_sample_data = preprocessed_data_dict['snow_depth']['snow_depth'][0].copy()
    nesosim_sample_data[:] = 1.0
    nesosim_gate_masks = sc.clip_to_shape_list(nesosim_sample_data, shapefiles_nesosim)

    gate_mask_dict = {}
    # todo for refactoring: reduce amount of nesting in dictionaries
    gate_mask_dict['Fram'] = nesosim_gate_masks[0]
    gate_mask_dict['Davis'] = nesosim_gate_masks[1]

elif dataset_name == 'smlg' or dataset_name == 'amsr2' or dataset_name == 'smlg-m2':
    shapefiles_smlg = sc.open_shapefiles_from_list(shapefile_names_smlg)
    # clip a single day to get a mask that's less computationally expensive to apply
    smlg_sample_data = preprocessed_data_dict['snow_depth']['snow_depth'][0].copy()
    smlg_sample_data[:]=1.0
    smlg_gate_masks = sc.clip_to_shape_list(smlg_sample_data, shapefiles_smlg) 

    gate_mask_dict = {}
    # todo for refactoring: reduce amount of nesting in dictionaries
    gate_mask_dict['Fram'] = smlg_gate_masks[0]
    gate_mask_dict['Davis'] = smlg_gate_masks[1]
elif dataset_name == 'legos_sam':
    shapefiles_legos = sc.open_shapefiles_from_list(shapefile_names_legos)
    legos_sample_data = preprocessed_data_dict['snow_depth']['snow_depth'][0].copy()
    legos_sample_data[:]=1.0
    # need to set spatial dims this way because otherwise it complains
    legos_sample_data = legos_sample_data.rio.set_spatial_dims('y','x')
    legos_sample_data = legos_sample_data.rio.write_crs("epsg:6931",inplace=True)
    legos_gate_masks = sc.clip_to_shape_list(legos_sample_data, shapefiles_legos) 
    gate_mask_dict = {}
    # todo for refactoring: reduce amount of nesting in dictionaries
    gate_mask_dict['Fram'] = legos_gate_masks[0]
    gate_mask_dict['Davis'] = legos_gate_masks[1]

elif dataset_name == 's3':
    shapefiles_s3= sc.open_shapefiles_from_list(shapefile_names_legos)
    # clip a single day to get a mask that's less computationally expensive to apply
    s3_sample_data = preprocessed_data_dict['snow_depth']['snow_depth'][0].copy()
    s3_sample_data[:]=1.0
    gate_mask_dict = {}
    gate_mask_dict['Fram'], gate_mask_dict['Davis'] = sc.clip_to_shape_list(s3_sample_data, shapefiles_s3)

elif dataset_name == 'mw99':
    shapefiles_mw99 = sc.open_shapefiles_from_list(shapefile_names_is2)
    mw99_sample_data = preprocessed_data_dict['snow_depth']['snow_depth'][0].copy()
    mw99_sample_data[:]=1.0
    gate_mask_dict = {}
    # todo for refactoring: reduce amount of nesting in dictionaries
    gate_mask_dict['Fram'],gate_mask_dict['Davis'] = sc.clip_to_shape_list(mw99_sample_data, shapefiles_mw99)


# %%
# nesosim_gate_masks[0].plot()

# %%
gate_mask_dict['Fram'].count().values

# %%
gate_mask_dict['Fram'].plot()

# %%
gate_mask_dict['Davis'].count().values

# %%
gate_mask_dict['Davis'].plot()

# %%

# %%
clipped_data_by_gate = clip_data_in_dict(preprocessed_data_dict, gate_mask_dict)
print('done clipping')



# %%

# %%
# filter out data where snow depth is less than zero
# need to preserve nan also

if dataset_name == 'legos_sam': # doing this for just this dataset but maybe should also apply to everything else as well actually
    for gate_name in clipped_data_by_gate.keys():
        # condition where snow depth less than zero
        depth_lt_zero_cond = clipped_data_by_gate[gate_name]['snow_depth']['snow_depth'] < 0
        # if snow depth is less than zero, set snow depth and salinity to be zero
        clipped_data_by_gate[gate_name]['snow_depth']['snow_depth'] = xr.where(depth_lt_zero_cond, 0, clipped_data_by_gate[gate_name]['snow_depth']['snow_depth'])
        clipped_data_by_gate[gate_name]['snow_salinity']['snow_salinity'] = xr.where(depth_lt_zero_cond, 0, clipped_data_by_gate[gate_name]['snow_salinity']['snow_salinity'])

        # adjust time indices to be at first day of month; this should only be done for legos_sam (which is monthly)

        clipped_data_by_gate[gate_name]['snow_depth']['time'] = pd.to_datetime(clipped_data_by_gate[gate_name]['snow_depth'].time).to_period('M').to_timestamp()
        clipped_data_by_gate[gate_name]['snow_density']['time'] = pd.to_datetime(clipped_data_by_gate[gate_name]['snow_density'].time).to_period('M').to_timestamp()
        clipped_data_by_gate[gate_name]['snow_salinity']['time'] = pd.to_datetime(clipped_data_by_gate[gate_name]['snow_salinity'].time).to_period('M').to_timestamp()

    

# %%
# save clipped data files

for gate_name in clipped_data_by_gate.keys():
    print(gate_name)
     # dataset_name is given at top of script
    for ds_name, ds in clipped_data_by_gate[gate_name].items():
        # print(ds)
        filename = 'shapefile_clipped_data/{}_{}_shapefile_clipped_{}.nc'.format(dataset_name, ds_name, gate_name)
        print(filename)
        sc.save_xr_comp(ds, filename)
print('done saving')


# %%
# plots for checking clipping regions, data orientation, etc

ival = 10
# ival  = 100

preprocessed_data_dict['snow_depth']['snow_depth'][ival].plot(cmap='viridis')
# plt.imshow(legos_gate_masks[1].values)
clipped_data_by_gate['Fram']['snow_depth']['snow_depth'][ival].plot(cmap='Reds')

clipped_data_by_gate['Davis']['snow_depth']['snow_depth'][ival].plot(cmap='Reds')

# plt.xlim(-4e-6,4e-6)
x0, y0 = plt.gca().get_xlim()

plt.show()

# %%
preprocessed_data_dict['snow_depth']['snow_depth'][ival].plot()

clipped_data_by_gate['Fram']['snow_depth']['snow_depth'][ival].plot(cmap='viridis')

clipped_data_by_gate['Davis']['snow_depth']['snow_depth'][ival].plot(cmap='viridis')
plt.show()

# %%
clipped_data_by_gate['Fram']['snow_depth']['snow_depth'][ival].plot()

clipped_data_by_gate['Davis']['snow_depth']['snow_depth'][ival].plot()
plt.show()

# %%
preprocessed_data_dict['snow_salinity']['snow_salinity'][ival].plot()
plt.show()


# %%
preprocessed_data_dict['myi_fraction']['myi_fraction'][10].plot()
plt.show()


# %%
preprocessed_data_dict['myi_fraction']['myi_fraction'][10].plot(cmap='Greys')


clipped_data_by_gate['Fram']['myi_fraction']['myi_fraction'][10].plot()

clipped_data_by_gate['Davis']['myi_fraction']['myi_fraction'][10].plot()
plt.show()

# %%

# %%
