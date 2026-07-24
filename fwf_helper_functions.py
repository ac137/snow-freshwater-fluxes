################################################################################
## fwf_helper_functions.py
## helper functions for performing calculations and
## plotting freshwater fluxes in freshwater_fluxes.ipynb
################################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import itertools

import seaborn as sb
import scipy.stats as st


def load_and_preprocess_ice_flux_data(flux_fn):
    flux_data = pd.read_csv(flux_fn)
    flux_data = flux_data.dropna(how='all')
    flux_data['Day'] = 1
    flux_data['Date'] = pd.to_datetime(flux_data[['Year','Month','Day']])
    flux_data.index = flux_data['Date']
    return flux_data


def load_data_into_dict(gate_name_list, data_name_list, file_dataset_list, date_index_name, fixed_dens=False,zero_sal=False):
    data_dict = {}
    for gate_name in gate_name_list:
        gate_data_dict = {}
        for i, data_name in enumerate(data_name_list):
            # these have MYI masking and also include uncertainties
            # load data
            extra_str = ''
            if fixed_dens:
                extra_str += '_dens293'
            if zero_sal:
                extra_str += '_0sal'
            data_current = pd.read_csv('gate_averages/{}{}_{}_freshwater_height_monthly_2016_onward_50km_centre_with_standard_dev.csv'.format(file_dataset_list[i],extra_str,
                                                                                                                                                         gate_name))
            # set index to a datetime index from the date column
            data_current.index = pd.to_datetime(data_current[date_index_name[i]])
            gate_data_dict[data_name] = data_current
        data_dict[gate_name] = gate_data_dict
    return(data_dict)

############## freshwater calculations

def calculate_ice_freshwater_flux(volume_flux):
    '''given sea ice volume flux volume_flux, calculate sea ice freshwater volume flux'''
    sal_ice = 4
    sal_ref = 34.8
    dens_ice = 901.3
    dens_seawater = 1023.9

    scale_factor = (1-sal_ice/sal_ref)*(dens_ice/dens_seawater)
    return volume_flux*scale_factor
    
def calc_snow_volume_flux(ice_volume_flux_df, snow_fw_dict, ice_area_flux, flux_col_name='freshwater_height'):
    '''calculate snow freshwater volume flux given snow freshwater height in snow_fw_dict, and ice_area_flux
    modifies ice_volume_flux_df in place to add snow freshwater for each snow product in snow_fw_dict'''
    for k,v in snow_fw_dict.items():
        ice_volume_flux_df[k] = v[flux_col_name]*ice_area_flux/1000

def calc_percent(x, y):
    '''calculate the percentage of x with respect to x+y when x is a df, y is a column'''
    return x*100/x.add(y,axis=0)

############# plotting functions
# function for fill_between (to simplify plotting)
def fill_between_err_df(df, err_df, ax, color,alpha=0.2):
    ''' given a dataframe df of data and dataframe err_df of data uncertainties (e.g. standard deviation),
     axes ax, and list of plot colours color,
     make fill-between plots for all columns in df. (df and err_df must have same columns)'''
    for i,c in enumerate(err_df.columns):
        ax.fill_between(df.index, df[c]-err_df[c], df[c]+err_df[c], alpha=alpha, color=color[i])

def fill_between_err_series(series, err_series, ax, color):
    ''' given a series of data and series err_series of data uncertainties (e.g. standard deviation),
     axes ax, and list of plot colours color,
     make fill-between plot (df and err_df must have same shape)'''
    # for i,c in enumerate(err_df.columns):
    ax.fill_between(series.index, series-err_series, series+err_series, alpha=0.2, color=color)

def plot_deseasonal_series(df_deseason, df_clim_std):
    ''' deseasonalized plot for single column of dataframe
    plots data and fill-between for a pandas series, with a grey horizontal line at zero'''
    df_deseason.plot()
    plt.gca().fill_between(df_clim_std.index, -df_clim_std, df_clim_std, alpha=0.2)
    
    plt.axhline(0,color='#bbbbbb')
    # leave out plt.show() so that other stuff can be added
    
def plot_deseasonal(df_deseason, df_clim_std):
    ''' deseasonalized plot for multiple columns of dataframe
    plots data and fill-between for all columns of a dataframe, with a grey horizontal line at zero'''
    df_deseason.plot()
    for c in df_clim_std.columns:
        plt.gca().fill_between(df_clim_std[c].index, -df_clim_std[c], df_clim_std[c], alpha=0.2)
    
    plt.axhline(0,color='#bbbbbb')


# for monthly plots
months_ordered = [10,11,12,1,2,3,4]
mon_names = ['Oct','Nov','Dec','Jan','Feb','Mar','Apr']

def plot_col_clim_strait(df,col_idx,data_name,ylabel,ax,date_lim='2021-04-01'):
    '''plot climatology by column'''

    box_plot_subset(df, 'Fram', date_lim, data_name, data_name + ' (Fram)', ylabel, ax[0,col_idx])
    box_plot_subset(df, 'Davis', date_lim, data_name, data_name + ' (Davis)', ylabel, ax[1,col_idx])

    ax[1,col_idx].set_xlabel('Month')
    ax[1,col_idx].set_xticklabels(mon_names)


def box_plot_subset(df, gate, date_lim, data_name, title, ylabel, ax):
    '''helper function to create box plots'''
    ts = df[gate].loc[:date_lim, data_name] 
    sb.boxplot(x=ts.index.month, y=ts,ax=ax,order=months_ordered)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def plot_trend_df(df, original_df, has_dt_index=False):
    ''' plot trends from trend dataframe df (output from trend_df_for_all_columns_in_df), 
    original_df is the df that was passed to trend_df_for_all_columns_in_df).
    if had_dt_index=True, assume a monthly datetime index (otherwise uses an integer index)
    creates a separate plot for each column'''
    
    for c in df.columns:
        xmin = df.loc['x_start',c]
        xmax = df.loc['x_end',c]
        b = df.loc['intercept',c]

        if has_dt_index:
            x = pd.date_range(xmin, xmax,freq='MS')
        else:
            x = np.arange(xmin, xmax+1)
        x_calc = np.arange(len(x))
        y = df.loc['slope',c]*x_calc + b
        y_high = df.loc['high_slope',c]*x_calc + b
        y_low = df.loc['low_slope',c]*x_calc + b
    
        # plt.plot(x, x_calc+b, color='black')
        plt.axhline(b,color='black')
        plt.plot(original_df.index, original_df[c],'.-',color='C0')
        plt.plot(x,y,color='C0')
        plt.plot(x,y_high,color='C2')
        plt.plot(x,y_low,color='C2')
        plt.title(c)
        plt.show()


############## time series operations

def clim_and_deseasonalize(df):
    ''' given dataframe df with a datetime index, calculate monthly
    climatology and climatological standard deviation (both same shape as df)
    and deseasonalize (subtract climatology from time series)
    returns climatology, climatological standard deviation, and deseasonalized
    time series
    '''
    df_clim = df.groupby(df.index.month).transform('mean')
    df_clim_std = df.groupby(df.index.month).transform('std')
    df_deseasonalized = df - df_clim
    return df_clim, df_clim_std, df_deseasonalized


def calc_annual_flux_data(df, sample_type='YS-OCT', count_threshold=10):
    '''given df, calculate time-resampled fluxes; sample_type determines
    frequency/start (defaults to year start in october). aggregated values
    with a count less than count_threshold are set to nan. returns resampled
    mean and standard deviation indexed by year'''
    # calculate annual fluxes for year start in october
    df_resampled = df.resample(sample_type)
    flux_totals = df_resampled.sum()
    flux_totals_sigma = df_resampled.std()
    flux_totals_count = df_resampled.count()

    flux_totals[flux_totals_count < count_threshold] = np.nan
    flux_totals_sigma[flux_totals_count < count_threshold] = np.nan
    
    flux_totals.index = flux_totals.index.year
    flux_totals_sigma.index = flux_totals_sigma.index.year

    return flux_totals, flux_totals_sigma

def calc_winter_flux_data(df, sample_type='YS-OCT'):
    '''given df, calculate time-resampled fluxes for october-april; sample_type determines
    frequency/start (defaults to year start in october). returns resampled
    mean and standard deviation indexed by year, along with the
    dataframe with only winter months (october-april) selected'''

    df_winter = df[df.index.month.isin([10,11,12,1,2,3,4])]

    # if we're calculating in winter, want to exclude anything with less than 1 month
    # so count threshold = 1 (equivalent to count == 0, since nothing has negative count)
    flux_totals, flux_sigma = calc_annual_flux_data(df_winter, count_threshold=1)

    return flux_totals, flux_sigma, df_winter


def trend_df_for_all_columns_in_df(df):
    ''' given dataframe df, calculate trends for all columns using theil-sen trend estimator
    return a dataframe of trends for each column with initial and final x values, slope, intercept,
    low_slope, and high_slope. if has_dt_index=True, '''

    trend_dict = {'slope':[],'intercept':[],'low_slope':[],'high_slope':[], 'x_start':[],'x_end':[]}
    key_vals = list(trend_dict.keys())
    col_list = df.columns

    x_index_nums = np.arange(len(df)) # numbering for x indices
    
    for c in col_list:
    
        # select values that are not nan
        vals_sel = df.loc[~df.loc[:,c].isna(),c]
        y = vals_sel.values
        # select from index in case some nan are skipped in the middle
        x = x_index_nums[~df.loc[:,c].isna()]

        trend_dict['x_start'].append(vals_sel.index[0])
        trend_dict['x_end'].append(vals_sel.index[-1])
    
        trend_vals = st.theilslopes(y, x)
    
        for i,k in enumerate(key_vals[:-2]):
            trend_dict[k].append(trend_vals[i])
    
    return pd.DataFrame(trend_dict,index=col_list).transpose()


##### uncertainty operations


def err_prop_percent(x, y, dx, dy):
    '''error propagation through percentage operation, x and dx are dataframes, y and dy are a single column value'''
    prefix = ((x.add(y,axis=0)**2)/100)**(-1) # 100/(x+y)^2

    term_1 = (x**2).mul(dy**2,axis=0) # x^2*dy^2
    term_2 = (dx**2).mul(y**2,axis=0) # y^2*dx^2
    
    return prefix*np.sqrt(term_1+term_2)

def percent_errprop_gate(data_df, sigma_df, gate_names, data_names, y_name):
    ''' percentage error propagation by gate; wrapper around error_prop_percent to make
    reindexing easier'''
    data_list = []
    for gate in gate_names:
        data_list.append(err_prop_percent(data_df[gate][data_names], data_df.loc[:,(gate,y_name)],
                                                  sigma_df[gate][data_names], sigma_df.loc[:,(gate,y_name)]))
    return data_list


############# data restructuring/aggregation 

def combine_fluxes_from_gates(fram_fw_v_flux, davis_fw_v_flux, fram_ice, davis_ice):
    ''' create dataframe that combines all the freshwater fluxes and also adds area flux, 
    for ease of plotting'''
    f_c = fram_fw_v_flux.copy()
    d_c = davis_fw_v_flux.copy()
    f_c['Ice area flux'] = fram_ice['IAF']
    d_c['Ice area flux'] = davis_ice['Area Flux']*-1
    return pd.concat([f_c, d_c], axis=1, keys=["Fram", "Davis"])