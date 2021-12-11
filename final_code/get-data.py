# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 09:48:05 2021

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

series = pd.read_csv('..\\final-data\\msa-series.csv')

keywords = ['All Employees: Education and Health Services',
 'All Employees: Financial Activities',
 'All Employees: Government',
 'All Employees: Leisure and Hospitality',
 'All Employees: Manufacturing',
 'Average Hourly Earnings of All Employees: Total Private',
 'Housing Inventory: Active Listing Count',
 'Housing Inventory: Median Days on Market',
 'Housing Inventory: Median Home Size in Square Feet',
 'Housing Inventory: Median Listing Price',
 'Housing Inventory: New Listing Count',
 'Per Capita Personal Income',
 'Resident Population',
 'Unemployment Rate']

fred_msa.save_all_series_data(api_key, series, keywords)
