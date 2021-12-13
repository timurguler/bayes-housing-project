# bayes-housing-project
Repository for Bayesian ML Final Group Project 2021

Contributors: Walter Coleman, Stephen Whetzel, Timur Guler

## Overview
This repository contains the code and output graphics for our implementation of vector autoregression (VAR) for predicting median housing price in real estate markets in New York, Atlanta, and Tulsa. The tool is somewhat universalizable as long as the input data matches the format used here and all key variables are updated in the relevant notebook. 

## Orientation
Readers will find all the relevant code in the "final_files>model files" folder. Within this folder is a module containing all of the functions necessary for the running of our VAR model. Feel free to explore this file on its own, however, it will be much more useful to look at the included python notebook "VAR.ipynb". This notebook contains a walkthrough of the implementation of the code in our module and will also output relevant visuals. This notebook can be run for any of the three cities available in our dataset "formatted_data.csv" by changing the name of the field "city" in the notebook and running all of the code again. 

**Note: there is a slightly counterintuitive convention used in our dataset. Variables with a "_#" suffix ("mean_housing_listing_price_4" for example) refer to a variable from a previous time stamp. The most recent time, t-1 being the largest number. For an order 5 VAR model, this means that "mean_housing_listing_price_4" refers to time t-1 and "mean_housing_listing_price_0" refers to time t-5.**

Change predictor mix or city of interest using the cell highlighted below within the VAR notebook. 


![image](https://user-images.githubusercontent.com/79474788/145843496-107d69a9-697a-4db9-b868-a62c8b641147.png)


## Images
In the images file you can find relevant images for all of our models. The files with the "_ci.png" suffix show the model predictions for 2019 with a band around our predictions for the 95% confidence interval as seen below for Atlanta. 


![atlanta_prediction_ci](https://user-images.githubusercontent.com/79474788/145843405-868d9cd6-32e2-4884-b026-e4b2a3852dde.png)
