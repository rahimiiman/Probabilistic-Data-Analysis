
import pandas as pd
import numpy as np
from dataanalysis import data_cleaning , DescriptiveStatistics


df = pd.read_excel('automobile_data_clean.xlsx')

des_stat_obj=DescriptiveStatistics(df)

#==============================
# boxplot 
#==============================
""" boxplot  is used to visualize the distribution of a continuous variable across different categories of a
categorical variable. Y axis is continuous variable and x axis is categorical variable
it is possible to group by multiple categorical variables to see the distribution of continuous variable
for example here we want to first see the distribution of price across different make
and then we want to see the distribution of price across different combination of aspiration and fuel-type
"""
des_stat_obj.boxplot([['make'],['aspiration','fuel-type']],['price','price'])

#==============================
# scatter plot with regression line
#==============================
""" scatter plot is used to visualize the relationship between two continuous variables.
it is possible to add a regression line to see the trend of the relationship
"""
des_stat_obj.scatter_with_regression(['city-mpg','highway-mpg','peak-rpm'],['price','price','price'])

#==============================
# histogram
#==============================
""" histogram is used to visualize the distribution of a continuous variable.
it is possible to specify the number of bins to control the granularity of the distribution
"""
des_stat_obj.histogram(['price','city-mpg','highway-mpg'], n_bin=20)


#==============================
# correlation matrix
#============================== 
""" correlation matrix is used to visualize the correlation (linear dependency) between multiple continuous variables.
it is possible to specify the method of correlation (pearson, spearman, kendall) and also to visualize the correlation matrix using a heatmap
"""
des_stat_obj.correlation_heatmap(columns='all', method='pearson', excel_filename="Descriptive_snapshots.xlsx", excel_sheet_name="Correlation")