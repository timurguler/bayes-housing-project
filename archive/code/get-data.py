# -*- coding: utf-8 -*-
"""
Extracting Historical Income Data by MSA using the FRED API

Created on Tue Dec  7 11:42:12 2021

@author: tgule
"""

import pandas as pd
import numpy as np
import os
import requests
import json
import time
import matplotlib.pyplot as plt
import fred_msa

api_key = 'a37b50cd27afbc3ce23a81ddc5541dec'

series = pd.read_csv('..\\fred-data\msa_series.csv')

keyword_list = ['Per Capita Personal Income', 'Resident Population', 'Unemployment Rate',
                'New Private Housing Units Authorized by Building Permits',
                'Regional Price Parities']

ny = series.query("city == 'New York'")
ny_housing = ny.title[ny.title.str.contains('Housing Inventory')]

housing_series = []
for s in ny_housing:
    if len(s.split(' in ')) > 2:
        housing_series.append(s.split(' in ')[0] + ' in ' + s.split(' in ')[1])
    else:
        housing_series.append(s.split(' in ')[0])

ny_employees = ny.title[ny.title.str.contains('All Employees')]

employees_series = []
for s in ny_employees:
    if len(s.split(' in ')) > 2:
        employees_series.append(s.split(' in ')[0] + ' in ' + s.split(' in ')[1])
    else:
        employees_series.append(s.split(' in ')[0])

keyword_list = keyword_list + housing_series + employees_series



fred_msa.save_all_series_data(api_key, series, keyword_list)

#fred_msa.save_series_data(api_key, series, 'Housing Inventory: New Listing Count')
