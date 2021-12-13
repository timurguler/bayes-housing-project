# -*- coding: utf-8 -*-
"""
Generating List of Available FRED MSA Series for Relevant MSAs


Created on Tue Dec  7 19:25:23 2021

@author: tgule
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import time
import fred_msa

api_key = 'a37b50cd27afbc3ce23a81ddc5541dec'
endpoint = 'https://api.stlouisfed.org/fred/series/categories'

state_ids = fred_msa.get_state_ids(api_key)

msa_ids = pd.DataFrame()
for state, state_id in state_ids.items():
    state_dict = fred_msa.get_msa_cats(api_key, state_id)
    states = fred_msa.extract_state(pd.Series(state_dict.keys()))
    state_df = pd.DataFrame({'msa' : list(state_dict.keys()), 'state' : states, 'ID' : list(state_dict.values())})
    msa_ids = msa_ids.append(state_df)
    time.sleep(3)

msa_ids['city'] = fred_msa.split_city(msa_ids.msa)
msa_ids = msa_ids[msa_ids.msa.str.contains('CMSA') == False]
msa_ids = msa_ids.drop_duplicates(subset = ['city', 'state']).reset_index(drop = True)

msas_to_use = pd.read_csv('..\\final-data\msas-to-use.csv', usecols=['msa'])

msas_to_use = pd.merge(msas_to_use, msa_ids,
         how='left', on='msa', validate='one_to_one')

series = pd.DataFrame()

for idx in range(len(msas_to_use)):
    msa_series = fred_msa.get_msa_series_list(msas_to_use.ID[idx], api_key)
    msa_series['msa'] = msas_to_use.msa[idx]
    msa_series['city'] = msas_to_use.city[idx]
    msa_series['state'] = msas_to_use.state[idx]
    
    series = series.append(msa_series).reset_index(drop=True)
    time.sleep(3)
    

series.to_csv('..\\final-data\msa-series.csv', index=False)
