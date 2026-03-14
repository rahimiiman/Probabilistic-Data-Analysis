"""
In this project we want to finad a regression model to predict the price of a car based on its features.
The dataset we are using is the automobile dataset which contains information about different cars and their prices.

Steps number is provided at the begining of each python file and you need to follow the steps in order to complete the project. 

1- we need to read the dataset and save it as csv file in local machine

2- check_database is done to get familiar with the dataset and to check if there are any missing values 
or outliers in the dataset. Also data types of each columns are checked to find if we need to handl formatting of columns

3- preprocessing : in preprocessing several steps are done:

        - Handdling missing values
        - converting categorical variables to numeric 
        - dtypes of columns are modified
        - normalization is done for some columns
        
4- descriptive statistics : the purpose of this step is to find most effective features for the regression model
we use scatter plot and person correlation to find more relevent features for the regression model.

5- model developement : in this step we want to find the best order of polynomial regression

first we search for the best order to figure out which polynomial order or maybe linear model we have to use
our criteria is R2 score and MSE and we use cross validation to find the best order of polynomial regression
after finding the best order we fit the model and then we test the model by randomly selecting some data points 
from the dataset and comparing the actual prices with the predicted prices.


"""
