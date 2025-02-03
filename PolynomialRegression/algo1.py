# In case if we have single feature

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from CSV file
data = pd.read_csv("C:/Users/mueez/OneDrive/Desktop/Machine/PolynomialRegression/file.csv")

# Ensure we're using 'Area' as input
X = data[['Area']].values  # Explicitly selecting 'Area'
y = data[['Price']].values  # Target variable is 'Price'

# Function to create polynomial features
def polynomial_features(X, degree):
    X_poly = np.ones((X.shape[0], 1)) # no of rows in x if 3 then create a an array of 3
    for d in range(1, degree + 1):  # range(start, stop) (1,3) loop will run 1 2 3 -- 3 times 
        X_poly = np.hstack((X_poly, X**d))
    return X_poly

# Set polynomial degree
degree = 2

# Transform X to polynomial features
X_poly = polynomial_features(X, degree)

# Compute theta using Normal Equation: θ = (X'X)^(-1) X'y
theta = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)

# Predict values
y_pred = X_poly.dot(theta)

# Plot results
plt.scatter(X, y, label="Actual Data", color="blue", alpha=0.6)
plt.plot(np.sort(X, axis=0), np.sort(y_pred, axis=0), label="Polynomial Regression", color="red", linewidth=2)
plt.xlabel("Area (sqft)")
plt.ylabel("Price")
plt.legend()
plt.title(f"Polynomial Regression (Degree {degree})")
plt.show()

# Print theta values
print("Optimized Theta values:")
print(theta)

# Predict the price for a new house with Area = 2500 sqft
new_area = 2500
new_area_poly = polynomial_features(np.array([[new_area]]), degree)  # Transform the new input like we did for training
predicted_price = new_area_poly.dot(theta)  # Use the trained model to predict price
print(f"Predicted price for a house with area {new_area} sqft: ${predicted_price[0][0]}")
