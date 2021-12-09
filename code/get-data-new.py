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

series = pd.read_csv('..\\fred-data-new\new-series-to-use.csv')

keywords = ['All Employees: Education and Health Services',
 'All Employees: Federal Government',
 'All Employees: Financial Activities',
 'All Employees: Goods Producing',
 'All Employees: Government',
 'All Employees: Government: Federal Government',
 'All Employees: Government: Local Government',
 'All Employees: Government: State Government',
 'All Employees: Information',
 'All Employees: Leisure and Hospitality',
 'All Employees: Local Government',
 'All Employees: Manufacturing',
 'All Employees: Mining, Logging, and Construction',
 'All Employees: Other Services',
 'All Employees: Private Service Providing',
 'All Employees: Professional and Business Services',
 'All Employees: Retail Trade',
 'All Employees: Service-Providing',
 'All Employees: State Government',
 'All Employees: Total Nonfarm',
 'All Employees: Total Private',
 'All Employees: Trade, Transportation, and Utilities',
 'Average Hourly Earnings of All Employees: Total Private',
 'Average Weekly Earnings of All Employees: Total Private',
 'Average Weekly Hours of All Employees: Total Private',
 'Housing Inventory: Active Listing Count',
 'Housing Inventory: Active Listing Count Month-Over-Month',
 'Housing Inventory: Active Listing Count Year-Over-Year',
 'Housing Inventory: Average Listing Price',
 'Housing Inventory: Average Listing Price Month-Over-Month',
 'Housing Inventory: Average Listing Price Year-Over-Year',
 'Housing Inventory: Median Days on Market',
 'Housing Inventory: Median Days on Market Month-Over-Month',
 'Housing Inventory: Median Days on Market Year-Over-Year',
 'Housing Inventory: Median Home Size in Square Feet',
 'Housing Inventory: Median Home Size in Square Feet Month-Over-Month',
 'Housing Inventory: Median Home Size in Square Feet Year-Over-Year',
 'Housing Inventory: Median Listing Price',
 'Housing Inventory: Median Listing Price Month-Over-Month',
 'Housing Inventory: Median Listing Price Year-Over-Year',
 'Housing Inventory: Median Listing Price per Square Feet',
 'Housing Inventory: Median Listing Price per Square Feet Month-Over-Month',
 'Housing Inventory: Median Listing Price per Square Feet Year-Over-Year',
 'Housing Inventory: New Listing Count',
 'Housing Inventory: New Listing Count Month-Over-Month',
 'Housing Inventory: New Listing Count Year-Over-Year',
 'Housing Inventory: Pending Listing Count',
 'Housing Inventory: Pending Listing Count Month-Over-Month',
 'Housing Inventory: Pending Listing Count Year-Over-Year',
 'Housing Inventory: Price Increased Count',
 'Housing Inventory: Price Increased Count Month-Over-Month',
 'Housing Inventory: Price Increased Count Year-Over-Year',
 'Housing Inventory: Price Reduced Count',
 'Housing Inventory: Price Reduced Count Month-Over-Month',
 'Housing Inventory: Price Reduced Count Year-Over-Year',
 'Per Capita Personal Income',
 'Resident Population',
 'Unemployment Rate']

fred_msa.save_all_series_data(api_key, series, pres_keywords)
