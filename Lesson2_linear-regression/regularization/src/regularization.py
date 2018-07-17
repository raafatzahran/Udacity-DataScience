# TODO: Add import statements
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('../data/data.csv')
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)

# Make and fit the linear regression model
linear_model = LinearRegression()
linear_model.fit(X,y)

# Retrieve and print out the coefficients from the regression model.
linear_reg_coef = linear_model.coef_
print(linear_reg_coef)