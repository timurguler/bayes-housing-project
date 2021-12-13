import pandas as pd
import numpy as np
import requests
import json
import os
import time
import fred_msa
from datetime import date
import regex as re
import seaborn as sns
import matplotlib.pyplot as plt


csvs_to_ignore = ['msa-series.csv', 'msas-to-use.csv'] # ignore files not directly from api pull

filepath = '..\\final-data'
data_files = set(os.listdir(filepath)).difference(set(csvs_to_ignore)) # list of data files to combine

col_names = fred_msa.get_new_colnames(data_files) # dict to rename columns from FRED series names
employee_cols = fred_msa.get_employee_cols(col_names) # list of columns dealing with employees in various sectors

data_dict = fred_msa.get_data_dict(data_files, filepath, col_names) # pull in data and metadata

id_vars = ['date', 'year', 'month', 'city', 'state'] # id variables - not predictors

data = fred_msa.convert_data_to_df(data_dict, ['date', 'year', 'month', 'city', 'state'], 'housing_median_listing_price') # convert to df

data = data.query("year < 2021") # 2020 and before

# columns to convert to float if not already
cols_to_leave = id_vars + ['housing_median_listing_price']

# convert to float for numeric operations
for col in data:
    if col not in cols_to_leave:
        data[col] = data[col].astype(float)
        
# columns to divide by population
to_divide = employee_cols + ['housing_active_listing_count', 'housing_new_listing_count']

# divide by population where applicable and normalize
# normalized = fred_msa.normalize_data(data, to_divide, cols_to_leave, ['housing_median_listing_price'])
# normalized.to_csv('..\\cleaned-data\\data-normalized.csv', index=False)

# get params for transformation
annual_cols = [k for k, v in data_dict.items() if v['frequency'] == 'Annual']
monthly_cols = [k for k, v in data_dict.items() if v['frequency'] == 'Monthly']
seasonal_cols = [col for col in data.columns if 'housing' in col]

monthly_order = 5
seasonal_order = annual_order = 1

# transform data and save
os.chdir('C:\\Users\\whetz\\Documents\\UVA MSDS\\Bayes Machine Learning\\Project\\bayes-housing-project\\cleaned-data')
data = pd.read_csv('data-no-transform.csv')
transformed = fred_msa.transform_data(data, id_vars, monthly_cols, seasonal_cols, annual_cols, monthly_order, seasonal_order, annual_order)
transformed.to_csv('..\\cleaned-data\\data_sw2.csv', index=False)