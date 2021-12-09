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
import regex as re

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

    filename = '..\\fred-data-new\\' + keyword.replace(':', ' -') + '.csv'

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