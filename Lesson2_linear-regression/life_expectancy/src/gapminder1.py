# TODO: Add import statements
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# Assign the dataframe to this variable.
# TODO: Load the data
path = '../data/bmi_and_life_expectancy.csv'
bmi_life_data = pd.read_csv(path, nrows=None, na_values=None)
X = bmi_life_data.iloc[:, :-1]
y = bmi_life_data.iloc[:, -1]
aux=bmi_life_data[['Life expectancy']]
aux1=bmi_life_data['Life expectancy']
# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)

print ("predict=",laos_life_exp)
