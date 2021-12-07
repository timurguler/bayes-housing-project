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

keyword = 'Per Capita Personal Income'

incomes = fred_msa.get_all_series(api_key, series, keyword)

filename = '..\\fred-data\\' + keyword + '.csv'

incomes.to_csv(filename, index=False)

