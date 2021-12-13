# bayes-housing-project
Repository for Bayesian ML Final Group Project 2021

Contributors: Walter Coleman, Stephen Whetzel, Timur Guler

## Overview
This repository contains the code and output graphics for our implementation of vector autoregression (VAR) for predicting median housing price in real estate markets in New York, Atlanta, and Tulsa. The tool is somewhat universalizable as long as the input data matches the format used here and all key variables are updated in the relevant notebook. 

## Orientation
Readers will find all the relevant code in the "final_files>model files" folder. Within this folder is a module containing all of the functions necessary for the running of our VAR model. Feel free to explore this file on its own, however, it will be much more useful to look at the included python notebook "VAR.ipynb". This notebook contains a walkthrough of the implementation of the code in our module and will also output relevant visuals. This notebook can be run for any of the three cities available in our dataset "formatted_data.csv" by changing the name of the field "city" in the notebook and running all of the code again. 
