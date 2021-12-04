# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:41:03 2021

@author: tgule
"""

import pandas as pd
import numpy as np
import os

pop_data = pd.read_csv("..\data\population-data.csv")
housing_prices = pd.read_csv("..\data\housing-prices.csv")
#rental_prices = pd.read_csv('..\data\rental-prices.csv')