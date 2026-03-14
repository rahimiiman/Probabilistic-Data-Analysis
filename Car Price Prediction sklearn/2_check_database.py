"""
This script checks the integrity of the data by reading the previously saved CSV file and generating a summary of the dataset.

using pd.describe(include='all') reflects some NaN values in the summary , for example for categorical column mean could not be calculated 
and it shows NaN in the summary, 

types of each columns are also added to the summary 

number of nulls is calculated and added to the summary to check for missing values in the dataset,




"""

import pandas as pd
import numpy as np
from dataanalysis import data_cleaning

# Read the data from the local CSV file
df = pd.read_csv('automobile_data.csv') 

analysis_obj=data_cleaning(df)

## summary of data base
analysis_obj.database_summary(output_name='automobile_data_summary.xlsx',missing_data_indicator='?')