"""

=============================
Missing values
=============================
at this part we need to deal with missing data shown in automobile_data_summary.csv file

There are different columns and the way we deal with each column is different

normalized-losses , bore , stroke ,horsepower ,peak-rpm we replace the missing data with average of the column

price : we need to drop the rows with missing price 

num-of-doors : the most frequent is 4 doors according to summary file , so we replace data with 4

=============================
Format
=============================
Some numeric columns are shown as object(string format in pandas) according to summary file
we need to convert this columns to either float or int

=============================
Adding new Metrics is needed
=============================
some times we need to add new features to database , this could be mapping current column to another one 
or calculating a new column using a combination of other columns and so on

the only thing we do here is to convert (miles per gallon)  to (litter per 100 km)

=============================
Data Normalization
=============================
"""

import pandas as pd
import numpy as np
from dataanalysis import data_cleaning


df = pd.read_csv('automobile_data.csv')
preproc_obj=data_cleaning(df)
#==============================
# Missing Data Handling
#==============================
# 1 , replace all missing data with np.nan 
# for example here missing data is shown by '?' , we need to change it to np.nan

preproc_obj.unify_missing_indicator(['?'])

# 2 drop entire row if price column is missing
preproc_obj.missing_data_handling('price',missing_data_indicator=np.nan,action='drop')

# 3 replace with average of column for these continuous value columns
selected_columns=['normalized-losses' , 'bore' , 'stroke' ,'horsepower' ,'peak-rpm' ]
preproc_obj.missing_data_handling(selected_columns,missing_data_indicator=np.nan,action='mean')

# 4 replace with most frequent value for discrete value columns
preproc_obj.missing_data_handling('num-of-doors',missing_data_indicator=np.nan,action='max')

#==============================
# New data base summary after cleaning 
#==============================
preproc_obj.database_summary(output_name='automobile_data_summary_after_missingdata.xlsx',missing_data_indicator='?')


#==============================
# Formatting
#==============================
selected_columns=["bore", "stroke","price","peak-rpm"]
preproc_obj.data_format_handling(selected_columns,dtype='float')
preproc_obj.data_format_handling('normalized-losses',dtype='int')

#==============================
# New Feature definition
#==============================

preproc_obj.df['city-L/100km'] = 235/df["city-mpg"]

preproc_obj.df["highway-mpg"] = 235/df["highway-mpg"]
preproc_obj.df=preproc_obj.df.rename(columns={'"highway-mpg"':'highway-L/100km'})

#==============================
# Handling_categorical
#==============================
# for two categorical column "fuel-type" and 'aspiration' we want to convert to bool using one-hot mapping
selected_columns=['fuel-type', 'aspiration']
preproc_obj.categorical_handling(selected_columns,prefix_sep='_',dtype='bool')

#==============================
# Normalization
#==============================
# save data before normalization
preproc_obj.df.to_excel('automobile_data_clean.xlsx',index=False)
selected_columns=['length','width','height']
preproc_obj.normalization(selected_columns,action='scaling')


#==============================
# export new summary file and  clean data
#==============================
preproc_obj.database_summary('automobile_data_summary_after_end_preprocessing.xlsx')
preproc_obj.df.to_csv('automobile_data_clean_normalized.csv')
    
