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
    