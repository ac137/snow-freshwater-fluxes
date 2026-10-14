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
## calc_freshwater_fluxes_from_clipped_data.ipynb
# script for calculating snow freshwater fluxes from data clipped using batch_clip_to_shapefile_intermediate.ipynb
# output plotted in freshwater_fluxes.ipynb
# .py file for this notebook generated using jupytext


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import os


import shapefile_clipping_helpers as sc

# %%
MASK_OUT_MYI = True

gate_list = ['Fram','Davis']
variable_list = ['snow_density', 'myi_fraction', 'snow_depth', 'snow_salinity']
# dataset_name = 'nesosim'
# dataset_name = 'smlg' #'smlg-m2'
# dataset_name = 'legos_sam'
dataset_name = 's3'
# dataset_name = 'amsr2'
# dataset_name = 'smlg-m2'


# USE_FIXED_DENSITY = True # if True, use prescribed single density
# USE_ZERO_SALINITY = False # if True, assume zero salinity

# USE_FIXED_DENSITY = False # if True, use prescribed single density
# USE_ZERO_SALINITY = True # if True, assume zero salinity

USE_FIXED_DENSITY = False # if True, use prescribed single density
USE_ZERO_SALINITY = False # if True, assume zero salinity

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

def adjust_salinity_with_myi_mask(data_dict, model_name):
    # modifies data_dict in place
    # adds variable 'salinity_adjusted' which is salinity zeroed out at MYI
    for gate_name, variable_to_data in data_dict.items():
        print(gate_name)
        # print(gate_to_model_name.keys())
        
        # print(variable_to_data.keys())

        if model_name not in ['legos_sam', 's3']:
            # if the model data is daily, need to forward fill for myi

            print('forward filling')
            # extend myi so that it's available daily and can be used as a daily mask
            myi_current = variable_to_data['myi_fraction'].resample(time="1D").ffill()
        else:
            # if the model data isn't daily, can just select myi fraction directly
            myi_current = variable_to_data['myi_fraction']
    
            # get the list of valid days in snow salinity
        if model_name == 'nesosim':
            day_list = variable_to_data['snow_salinity'].day # list of valid days in snow salinity
        elif model_name in ['smlg','smlg-m2','legos_sam','s3', 'mw99']:
            day_list = variable_to_data['snow_salinity'].time # list of valid days in snow salinity
        elif model_name == 'amsr2':
            day_list = variable_to_data['snow_salinity'].date

        # get strings denoting the zeroth and final days
        day_zero_string = day_list[0].dt.date.astype('str').values
        final_day_string = day_list[-1].dt.date.astype('str').values
        
        print('selecting dates')
    
        # amsr2 is discontinuous; need to select specific dates since some are missing
        if model_name=='amsr2':
            # select specific dates for amsr2 (some missing, not continuous)
            dates_amsr2 = variable_to_data['snow_depth'].date.loc[day_zero_string:final_day_string].dt.date.astype(str).values
            # need to adjust time variable for myi to enable this selection because cftime is finicky
            myi_current['time'] = pd.to_datetime(myi_current.time.dt.floor('D').astype(str).values)
            myi_current = myi_current['myi_fraction'].loc[dates_amsr2]
        elif model_name in ['legos_sam', 's3','mw99']:
            dates_data = variable_to_data['snow_depth'].time.loc[day_zero_string:final_day_string].dt.date.astype(str).values
            myi_current['time'] = pd.to_datetime(myi_current.time.dt.floor('D').astype(str).values)
            myi_current = myi_current['myi_fraction'].loc[dates_data]
            # need to adjust time variable for myi to enable this selection because cftime is finicky
        # elif model_name==mw99:
        #     dates_data = variable_to_data['snow_depth'].time.loc[day_zero_string:'2024-12'].dt.date.astype(str).values
        #     myi_current['time'] = pd.to_datetime(myi_current.time.dt.floor('D').astype(str).values)
        #     myi_current = myi_current['myi_fraction'].loc[dates_data]
        else:
        
            myi_current = myi_current['myi_fraction'].loc[day_zero_string:final_day_string]
            
        # if it's nesosim leave out the summer months

        if model_name == 'nesosim':
        # select only non-summer months for nesosim
            myi_current = myi_current.sel(time=myi_current.time.dt.month.isin([9, 10, 11, 12, 1,2,3,4]))
        # apply the myi filter
        # assign new adjusted salinity
        print('applying salinity mask')
        # print(variable_to_data.keys())
        
        data_dict[gate_name]['salinity_adjusted'] = apply_mask_to_salinity(variable_to_data['snow_salinity'], 
                                                                         myi_current)
    # return clipped_data_by_gate

def calculate_salinity_height_for_dict(data_dict, model_name, use_fixed_density=False, fixed_density=293., use_zero_salinity=False):
    '''modifies data_dict in place
    adds 'freshwater_height' (effective water-equivalent height of snow freshwater),
    'freshwater_height_monthly_grid' (monthly mean of above),
    'freshwater_height_monthly' (spatial average of monthly freshwater height)
    variables ending in _sd are the corresponding standard deviation
    
    '''
    keyword_vars = {'nesosim':'day','amsr2':'date','smlg':'time','smlg-m2':'time', 'legos_sam':'time','s3':'time', 'mw99':'time'}
    kw = {'{}'.format(keyword_vars[model_name]): 'MS'}
    for gate_name, variable_to_data in data_dict.items():

        # for model_name, variable_to_data in gate_to_model_name.items():
        if use_fixed_density:
            # option to use a prescribed constant density value
            density_to_use = fixed_density
            print('using fixed density value {}'.format(fixed_density))
        else:
            density_to_use = variable_to_data['snow_density']['snow_density']
        if use_zero_salinity:
            # option for zero snow salinity
            salinity_to_use = 0
            print('using zero salinity')
            # could probably streamline zero salinity more but leaving as-is for now
        else:
            salinity_to_use = variable_to_data['salinity_adjusted']['snow_salinity']

        variable_to_data['freshwater_height'] = calc_salinity_height(variable_to_data['snow_depth']['snow_depth'], 
                                                                salinity_to_use, 
                                                                    density_to_use)
        
        # try kw argument unpacking here instead of going by model name?. would replace if statements with a dict lookup or something here I guess
        # keyword_var{'nesosim':'day'}; kw = {'{}'.format(keyword_var[model_name]): 'MS'}; .resample(**kw)

        variable_to_data['freshwater_height_monthly_grid'] = variable_to_data['freshwater_height'].resample(**kw).mean()
        # standard deviation of monthly mean
        variable_to_data['freshwater_height_monthly_grid_sd'] = variable_to_data['freshwater_height'].resample(**kw).std()

        # add names to these quantities to enable conversion to df
        variable_to_data['freshwater_height_monthly_grid'].name='freshwater_height'
        variable_to_data['freshwater_height_monthly_grid_sd'].name='freshwater_height'

        # gate-averaging
        monthly_fwh_df = variable_to_data['freshwater_height_monthly_grid'].mean(dim=['x','y']).to_dataframe()
        data_col = [monthly_fwh_df.columns[-1]] # data column name
        
        monthly_fwh_df = monthly_fwh_df[data_col] # select specifically just the data column

        # standard deviation over gate
        monthly_fwh_df_std = variable_to_data['freshwater_height_monthly_grid'].std(dim=['x','y']).to_dataframe()
        monthly_fwh_df['spatial_std'] = monthly_fwh_df_std[data_col] # add to df as another column

        
        # propagate monthly standard deviation through gate average (sqrt(sum(value**2))/N)
        # count valid data (to provide N in the formula above)
        fw_count = variable_to_data['freshwater_height_monthly_grid'].count(dim=['x','y'])
        monthly_fwh_df_std_propagated = ((variable_to_data['freshwater_height_monthly_grid_sd']**2).sum(dim=['x','y'])**(1/2)/fw_count).to_dataframe()
        monthly_fwh_df['temporal_std'] = monthly_fwh_df_std_propagated[data_col] # add to df as another column

        
        variable_to_data['freshwater_height_monthly'] = monthly_fwh_df # add as a single value


# %%
# load data into dictionary
clipped_data_dict = sc.load_clipped_data_to_dict(dataset_name, gate_list, variable_list)

# clipped_data_dict

# %%
# apply MYI mask to adjust salinity
adjust_salinity_with_myi_mask(clipped_data_dict, dataset_name)
print('done')

# %%
calculate_salinity_height_for_dict(clipped_data_dict, dataset_name, use_fixed_density=USE_FIXED_DENSITY,use_zero_salinity=USE_ZERO_SALINITY)
print('done')

# %%

# %%
clipped_data_dict['Fram']['freshwater_height_monthly'].plot()
plt.show()

# %%
clipped_data_dict['Davis']['freshwater_height_monthly'].plot()
plt.show()

# %%
# can save the data now

file_path = 'gate_averages/'
buffer_size = 50
quantity_name = 'freshwater_height_monthly'
# quantity_name = 
extra_str = '2016_onward_{}km_centre_with_standard_dev'.format(buffer_size)

if USE_FIXED_DENSITY:
    dataset_name += '_dens293'
if USE_ZERO_SALINITY:
    dataset_name += '_0sal'

for gate, data_vals in clipped_data_dict.items():
    # print(gate)
    # # print(model_to_value.keys())

    print(dataset_name)
    print(data_vals.keys())
    fw_height_df = data_vals[quantity_name]
    # print(fw_height_df)
    # save output

    filename = '{}{}_{}_{}_{}.csv'.format(file_path, dataset_name, gate, quantity_name, extra_str)
    print(filename)
    fw_height_df.to_csv(filename)

print('data saved')

# %%
print('done')

# %%
filename

# %%

# %%
