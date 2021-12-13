# -*- coding: utf-8 -*-
"""
FRED API MSA Functions

This module contains a set of functions to 

Created on Tue Dec  7 11:21:03 2021

@author: tgule
"""

import pandas as pd
import numpy as np
import json
import os
import requests
import time
from datetime import date
import regex as re
import matplotlib.pyplot as plt
import seaborn as sns


########### Section 1 - Extracting City and State from MSA Names

def split_city(city_list):
    '''
    Takes out city name and other listed cities to isolate main city name fro MSA name
    '''
    return [str.replace(re.split('-|/', s[0])[0], ' County', '') for s in city_list.str.split(',')]

def extract_state(city_list):
    '''
    Extracts main state from list of city names
    '''
    return [str.strip(str.split(s[1], sep='-')[0]) for s in city_list.str.split(',')]


########## Section 2 - Pulling Data with the FRED API

def get_state_ids(api_key):
    
    '''
    Gets a list of state ids from the FRED API. The parent category for all states has id '27281', 
    so getting all children returns the individual state ids
    '''
    
    params_location = {
    'api_key' : api_key,
    'category_id' : '27281',
    'file_type': 'json'
    }

    endpoint_location = 'https://api.stlouisfed.org/fred/category/children'
    
    response_location = requests.get(endpoint_location, params=params_location)
    state_ids = {s['name'] : s['id'] for s in json.loads(response_location.text)['categories']}
    
    return state_ids

def get_msa_cats(api_key, state_id):
    '''
    For a given FRED state id, get a list of FRED IDs for all Metro Statistical Areas in that state
    '''

    params_state = {
        'api_key' : api_key,
        'category_id' : state_id,
        'file_type': 'json'
    }

    endpoint_state = 'https://api.stlouisfed.org/fred/category/children'
    response_state = requests.get(endpoint_state, params=params_state)
    
    state_msas = {} # instantiate dictionary for MSAs to enure valid return statement
    
    cats = json.loads(response_state.text)['categories']
    
    if len(cats) > 0: # not all states/territories have MSAs
        
        # "categories call returns both counties and MSAs; we only want MSAs
        msa_cat = [s['id'] for s in json.loads(response_state.text)['categories']][1] 

        params_msacat = {
                'api_key' : api_key,
                'category_id' : msa_cat,
                'file_type': 'json'
            }

        endpoint_msacat = 'https://api.stlouisfed.org/fred/category/children'

        response_msacat = requests.get(endpoint_msacat, params=params_msacat)

        state_msas = {s['name'] : s['id'] for s in json.loads(response_msacat.text)['categories']}
    
    return state_msas

def get_msa_series_list(cat, api_key):
    '''
    Get a list of all available FRED series for a particular MSA

    Parameters
    ----------
    cat : number cast as string
        FRED category of MSA
    api_key : string
        API key

    Returns
    -------
    msa_series : dataframe
        Table of all available FRED Series and descriptions for given MSA

    '''
    
    params = {
    'api_key' : api_key,
    'category_id' : cat,
    'file_type': 'json'
    }
    
    endpoint = 'https://api.stlouisfed.org/fred/category/series'

    response = requests.get(endpoint, params=params)
    
    msa_series = pd.DataFrame()
    
    try:
        msa_series = pd.DataFrame(json.loads(response.text)['seriess'])
        #msa_series = msa_series[[any(key in s for key in keywords) for s in msa_series.title]]
    except:
        pass
    
    return msa_series
    #v[any(keyword in string for substring in substring_list)]
    
def get_series(series_id, api_key):
    '''
    

    Parameters
    ----------
    series_id : string
        the FRED series id
    api_key : string
        FRED API key

    Returns
    -------
    observations : dataframe of observations
        dataframe of observations

    '''
    params = {
        'api_key' : api_key,
        'series_id' : series_id,
        'file_type': 'json'
    }
    
    endpoint = 'https://api.stlouisfed.org/fred/series/observations'
    
    response = requests.get(endpoint, params=params)
    observations = pd.DataFrame(json.loads(response.text)['observations'])
    
    return observations

def get_all_series(api_key, series_df, keyword):
    '''

    Parameters
    ----------
    api_key : string
        FRED API KEY.
    series_df : dataframe
        dataframe of FRED MSA series, including columns "title", "id", "state", and "msa"
    keyword : string
        name of overall series (e.g. "Per Capita Personal Income")

    Returns
    -------
    output : dataframe
        dataframe with data, value, state, and msa for ALL FRED MSA data series matching the keyword

    '''
    reg_keyword = r'^{}'.format(keyword)
    subseries = series_df[series_df.title.str.match(reg_keyword)==True].reset_index(drop=True)
    subseries = subseries.sort_values(['msa', 'frequency', 'seasonal_adjustment'], ascending=[True, False, False])
    subseries = subseries.groupby(['msa']).head(1).reset_index()
    #print(subseries[['city', 'frequency', 'seasonal_adjustment']])
    output= pd.DataFrame()
    
    for idx in range(len(subseries)):
        series_output = get_series(subseries.id[idx], api_key)
        series_output['state'] = subseries.state[idx]
        series_output['msa'] = subseries.msa[idx]
        series_output['city'] = subseries.city[idx]
        series_output['title'] = subseries.title[idx]
        series_output['id'] = subseries.id[idx]
        series_output['frequency'] = subseries.frequency[idx]
        series_output['seasonal_adjustment'] = subseries.seasonal_adjustment[idx]
        
        series_output = series_output.rename(columns={'value' : keyword})
        series_output = series_output.drop(columns=['realtime_start', 'realtime_end'])
        
        output = output.append(series_output)
        time.sleep(3)
        
    return output
    
def save_series_data(api_key, series_df, keyword):
    '''
    Parameters
    ----------
    api_key : string
        FRED API KEY.
    series_df : dataframe
        dataframe of FRED MSA series, including columns "title", "id", "state", and "msa"
    keyword : string
        name of overall series (e.g. "Per Capita Personal Income")

    Returns
    -------
    saves a csv file with data to fred-data folder
    '''

    df = get_all_series(api_key, series_df, keyword)

    filename = '..\\final-data\\' + keyword.replace(':', ' -') + '.csv'

    df.to_csv(filename, index=False)
    
    
def save_all_series_data(api_key, series_df, keyword_list):
    '''

    Parameters
    ----------
    api_key : string
        FRED API key
    series_df : dataframe
        dataframe of FRED MSA series, including columns "title", "id", "state", and "msa"
    keyword_list : list of strings
        DESCRIPTION.

    Returns
    -------
    saves csv file with data to fred-data folder for all keywords in keyword list

    '''
    
    for keyword in keyword_list:
        save_series_data(api_key, series_df, keyword)
        
        
######### Section 3 - Combining and Transforming FRED data

def get_new_colnames(filenames):
    '''
    Creates a dictionary of new column names from FRED series names

    Parameters
    ----------
    filenames : set
        set of filenames with relevant data series

    Returns
    -------
    new_col_names : dictionary
        mapping of old names to new names

    '''
    new_col_names = {} # rename columns for ease of use
    for k in filenames:

        if re.match(r'^All Employees', k): # proportion of pop in various industries
            field = str.lower(re.split(' ', re.split('-', k)[1])[1].strip('.csv'))
            new_col_names[k] = field

        elif 'Hourly Earnings' in k:
            new_col_names[k] = 'hourly_earnings'

        elif re.match(r'^Housing Inventory', k): # 
            new_col_names[k] = 'housing_' + str.lower(re.split('- ', k)[1].replace(' ', '_').strip('.csv'))

        else:
            new_col_names[k] = str.lower(k.replace(' ', '_').strip('.csv'))
        
    return new_col_names


def get_employee_cols(col_dict):
    '''
    Extracts column names where series relates to number of employees in a sector

    Parameters
    ----------
    col_dict : dict
        results from get_new_colnames function

    Returns
    -------
    employee_cols : list
        list of col names dealing with employees in sector

    '''
    employee_cols = [v for k, v in col_dict.items() if re.match(r'^All Employees', k)]
    return employee_cols

def import_data_file(file, filepath, col_dict):
    '''
    Imports data and metadata relating to FRED series, saved with specific formats via save_all_series function

    Parameters
    ----------
    file : string
        name of file
        
    filepath : string
        path where files stored
        
    col_dict : dictionary
        crosswalk mapping of FRED series names to col names

    Returns
    -------
    data_and_meta : dict
        dictionary w dataframe of data and some metadata fields

    '''
    
    data = pd.read_csv(os.path.join(filepath, file))
    frequency = data.frequency[0]
    
    data.date = pd.to_datetime(data.date)
    data['year'] = [d.year for d in data.date]
    
    data.columns.values[1] = col_dict[file]
    
    if frequency == 'Monthly':
        data['month'] = [d.month for d in data.date]
        
    to_drop = ['msa', 'title', 'id', 'frequency', 'seasonal_adjustment']
    data_and_meta = {'data' : data.drop(columns=to_drop), 'frequency' : data.frequency[0]}
    
    return data_and_meta


def get_data_dict(file_list, filepath, col_dict):
    '''
    creates dict of data + metadata for all FRED series in data folder

    Parameters
    ----------
    file_list : set
        set of file names
        
    filepath : string
        path where files stored
        
    col_dict : dictionary
        crosswalk mapping of FRED series names to col names

    Returns
    -------
    data_dict : dict
        dict of data + metadata for all FRED series in data folder
    
    '''
    
    data_dict = {}

    for file in file_list:
        data_dict[col_dict[file]] = import_data_file(file, filepath, col_dict)
        
    return data_dict

def convert_data_to_df(data_dict, id_vars, target_var):
    '''
    converts data dict from get_data_dict function into df

    Parameters
    ----------
    data_dict : dict
        dictionary of data and metadata from get_data_dict
        
    id_vars : list of columns which contain key variables
        
    target_var : string
        target variable for later analysis

    Returns
    -------
    dataset : df
        df of all variables needed for analysis
    
    '''
    dataset = data_dict[target_var]['data'][id_vars + [target_var]]

    for k, v in data_dict.items():
        if k != target_var:
            if v['frequency'] == 'Monthly':
                dataset = pd.merge(dataset, v['data'].drop(columns=['date']), how = 'left', on = ['year', 'month', 'city', 'state'], validate = 'one_to_one')
            else:
                dataset = pd.merge(dataset, v['data'].drop(columns=['date']), how = 'left', on = ['year', 'city', 'state'], validate = 'many_to_one')
    
    return dataset

def normalize_data(df, cols_to_divide, cols_not_normalized, cols_log_trans):
    
    '''
    performs normalization, conversion to per capita measures, and log transforms

    Parameters
    ----------
    df : df
        df of all vars needed for analysis
        
    cols_to_divide : list of cols to convert to per capita
    
    cols_not_normalized : list of columns to NOT normalize (all rest will be normalized)
    
    cols_log_trans : list of columns to log transform
        
    Returns
    -------
    normalized : df
        df post normalization and prep procedures
    '''
    
    #convert to per capita
    for col in cols_to_divide:
        df[col] = df[col]/df.resident_population
    
    normalized = df.drop(columns = cols_not_normalized).transform(lambda x: (x - x.mean()) / x.std())

    normalized = pd.concat([df[cols_not_normalized], normalized], axis=1)

    for col in cols_log_trans:
        normalized[col] = np.log(normalized[col])
    
    return normalized

def transform_data(df, id_vars, monthly_cols, seasonal_cols, annual_cols, monthly_order, seasonal_order, annual_order):
    
    '''
    converts data to format needed for Variational autoregression

    Parameters
    ----------
    df : df
        df of all vars needed for analysis (post normalization)
        
    monthly_cols : list of cols that report monthly
    
    seasonal_cols : list of cols with seasonality
    
    annual_cols : list of cols that report annually
    
    monthly, seasonal, annual order : number of periods to look back for monthly and annual trends as well as seasonality (12 months previously)
    
        
    Returns
    -------
    transformed : df
        df with current and previous values for variables
    
    '''
    
    output_df = df.copy()

    seasonal = output_df[seasonal_cols]
    monthly = output_df[monthly_cols]
    annual = output_df[annual_cols]

    for i in range(1, monthly_order+1):
        shifted = monthly.shift(i).rename(columns={col : col + '_' + str(monthly_order - i) for col in monthly.columns})
        output_df = pd.concat([output_df, shifted], axis=1)

    for i in range(1, seasonal_order+1):
        shifted = seasonal.shift(i*12).rename(columns={col : col + '_s' + str(seasonal_order-i) for col in seasonal.columns})
        output_df = pd.concat([output_df, shifted], axis=1)

    for i in range(1, annual_order+1):
        shifted = annual.shift(i*12).rename(columns={col : col + '_' + str(annual_order-i) for col in annual.columns})
        output_df = pd.concat([output_df, shifted], axis=1)
        
    return output_df

def time_plot(df, id_vars, cols_to_ignore):
    
    '''
    given a dataframe, key variables, and columns to leave out, plots a grid of time trends
    for each variable by city, with a black dashed line for the start of COVID
    '''
    trim_df = df.drop(columns=cols_to_ignore)
    melted = pd.melt(trim_df, id_vars=id_vars)
    
    g = sns.FacetGrid(melted, col = 'variable', hue='city', col_wrap=3, height=4, aspect=1)
    g.map(plt.plot, 'date', 'value', linewidth=2)
    #g.map(plt.axvline(x='observed'))
    g.set_titles('{col_name}', fontsize=16)
    g.set_axis_labels('Date', 'Normalized/Log Value', fontsize=16)
    g.fig.subplots_adjust(top=.9)
    g.fig.suptitle('Time Trends for Predictor Variables', fontsize=16)
    g.add_legend()

    dates = [date(2020, 3, 11)]*len(melted.variable.unique())

    for ax, pos in zip(g.axes.flat, dates):
        ax.axvline(x=pos, color='black', linestyle='--')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

def median_scatter(df, id_vars, target_var, **cols_to_ignore):
    '''
    given a dataframe, key variables, target vars, and optionally columns to leave out, 
    plots a grid of scatterplots of target vs predictor for each predictor, by city
    '''
    trim_df = df.drop(columns=cols_to_ignore)
    melted = pd.melt(trim_df, id_vars = id_vars + [target_var])

    g = sns.FacetGrid(melted, col = 'variable', hue='city', col_wrap=3, height=4, aspect=1)
    g.map(plt.scatter, 'value', target_var, alpha=0.2)
    #g.map(plt.axvline(x='observed'))
    g.set_titles('{col_name}', fontsize=16)
    g.set_axis_labels('Value', 'Median Housing Price', fontsize=16)
    g.fig.subplots_adjust(top=.9)
    g.fig.suptitle('Scatter for Predictor Variables', fontsize=16)
    g.add_legend()