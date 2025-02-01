# Here data set is from 1 - 10,000 so we are going to use closed system 
# Formula y = mx + b

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/mueez/OneDrive/Desktop/Machine/SimpleLinearRegression/file.csv')
x = df['Area'].values # exctracted the col of area 
y = df['Price'].values


# to apply formula y = mx + b 1st we are going to find the value of m and b 


# formula for m 
N = len(x)
sum_x_y = np.sum(x * y)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x_square = np.sum(x ** 2)

m = (N * sum_x_y - sum_x * sum_y)/(N * sum_x_square - (sum_x ** 2))

# formula for b 
b = (sum_y - m * sum_x ) / N
print(b)

X = float(input("Enter the Area of house: ")) # X is the area (input) that we are gonna give and in result the model is going to predict
y = m * X  +b
print(f"The price is: ${y}")


# The slight difference in price predictions is expected in linear regression because it's modeling the general trend of the data rather than exact values for every data point.
# If you need better accuracy, check the fit of the model and ensure the data doesn't have too much noise or non-linearity. You can also try polynomial regression for a more accurate fit.