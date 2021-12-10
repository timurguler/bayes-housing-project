# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:54:26 2021

@author: whetz
"""

import pandas as pd
import numpy as np
import pymc3 as pm
import graphviz
import arviz as az
import seaborn as sns
from scipy.stats import norm


def read_data(predictors, path="fred-data-pres\\pres-data.csv"):
    df = pd.read_csv(path, index_col=0).reset_index().query("year < 2020")
    df = df.rename(
        {
            "Housing Inventory: Median Listing Price": "med_housing",
            "Unemployment Rate": "unemployment",
            "All Employees: Federal Government	Housing Inventory: Median Home Size in ": "govt_employees",
            "Housing Inventory: Median Days on Market": "housing_mkt_days",
            "Housing Inventory: New Listing Count": "housing_listings",
            "All Employees: Financial Activities": "financial_act",
            "All Employees: Education and Health Services": "educ_health",
            "All Employees: Federal Government": "fed_employees",
            "Resident Population": "population",
            "Per Capita Personal Income": "income",
            "Housing Inventory: Median Home Size in Square Feet": "home_size",
        },
        axis=1,
    )

    df.insert(0, "year_month", df.year + df.month / 100)

    id_cols = [
        'year_month',
        'year',
        'month',
        'city',
        'state'
        ]
    id_cols.extend(predictors)
    df = df[id_cols]

    return df


def generate_time_data(df, city, order, predictors):

  df = df.query(F"city== '{city}'").reset_index(drop=True)

  df_input = pd.DataFrame()

  year_months = []

  for period in range(order, len(df)):
    year_month = df.year_month.loc[period]
    # print(year_month)
    df2 = df.loc[period-order:period-1]
    # print(df2)

    df2['step'] = [i for i in range(order)]
    df_pivot = df2.pivot(index='city', columns='step', values=predictors).reset_index()
    df_pivot.columns = [col[0]+"_"+str(col[1]) for col in df_pivot.columns]
    # print(df_pivot)

    # Get relevant outputs
    query = F"year_month == {year_month}"
    for col in predictors:
      df_pivot[col] = df.query(query).reset_index(drop=True)[col].iloc[0]

    df_input = pd.concat([df_input, df_pivot])
    year_months.append(year_month)

  # print(df_input)

  df_input.insert(1, 'year_month',year_months)
  df_input = df_input.reset_index(drop=True)
  # df_input.columns=[str(col[0]) + str(col[1]) for col in df_input.columns.values]
  return df_input