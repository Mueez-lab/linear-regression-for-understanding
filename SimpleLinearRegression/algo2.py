# Now imagine that dataset from 100,000 to 1M
# So we are going to use gradient descent 

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/mueez/OneDrive/Desktop/Machine/SimpleLinearRegression/file.csv')
x = df['Area'].values
y = df['Price'].values


# Normalization 

x = (x-np.mean(x))/ np.std(x)

# now calculating the value of m and v
N = len(x)
alpha = 0.001
iterations = 1000
m = 0
b = 0

# Gradient Descent 

for i in range(iterations):
    y_pred = m*x + b
    gradient_m = (1/N) * np.sum((y_pred-y)*m) # This is ∂J/∂m
    gradient_b = (1/N) * np.sum(y_pred-y) # This is ∂J/∂b
    
    m -= alpha * gradient_m
    b -= alpha * gradient_b
    # optional just for tracking purpose 
    if i % 100 == 0:
        cost = (1/(2*N))* np.sum((y_pred - y)**2)
        print(f"Iteration {i}: Cost = {cost:.4f}, m = {m:.4f}, b = {b:.4f}") 

# Print the final m and b after the training is complete
print(f"Final slope (m): {m}")
print(f"Final intercept (b): {b}")

X = float(input("Enter the Area of house (in square feet): "))
y_pred = m*X+b

print(f"The price is ${y_pred}")