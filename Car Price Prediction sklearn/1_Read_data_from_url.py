"""
This script demonstrates how to read data from a URL using pandas. 
It uses the `read_csv` function to load data directly from a specified URL, 
which is particularly useful for accessing datasets hosted online without needing to download them first.


"""

import pandas as pd

url= "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

# Read the data from the URL
df = pd.read_csv(url, header=None)   # CSV file in server does not have header, so we set header=None

# Display the first few rows of the data
print(df.head())


header = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors',
          'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders',
          'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

#assigning the column names to the data
df.columns = header

#save the data to a local CSV file
df.to_csv('automobile_data.csv', index=False)  # index=False to avoid writing row numbers to the file