import pandas as pd
import numpy as np
from dataanalysis import data_cleaning , DescriptiveStatistics , regression_analysis


df = pd.read_excel('automobile_data_clean.xlsx')

reg_analysis_obj = regression_analysis(df)

#==============================
# order check
#==============================
feature_cols = ['curb-weight','engine-size','horsepower','city-mpg','highway-mpg','width','length','height','wheel-base']
target_col = 'price'
best_order = reg_analysis_obj.best_order_finding(feature_cols, target_col, max_order=3,
                                    criterion='r2',method='cross_val',kfold=3,
                                    excel_filename="Descriptive_snapshots.xlsx", excel_sheet_name="Best_order_finding")


fitted_model, r2, mse = reg_analysis_obj.regression_fitting(feature_cols, target_col, order=best_order)

                                    
                                    
##==============================
# prediction
#==============================

#randomly selecting some data points from the dataset to test the prediction
n=5
sample_data = df.sample(n)
new_data = sample_data[feature_cols]
actual_prices = sample_data[target_col]
predicted_prices = fitted_model.predict(new_data)
print("Actual prices:\n", actual_prices.values)
print("Predicted prices:\n", predicted_prices)
