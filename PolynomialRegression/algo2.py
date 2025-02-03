import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("C:/Users/mueez/OneDrive/Desktop/Machine/PolynomialRegression/multi_feature_file2.csv")

# Select multiple features (e.g., Area, Bedrooms, Floors)
X = data[['Size (sqft)', 'Bedrooms', 'Bathrooms', 'House Age (years)']].values # 2d
y = data[['Price']].values  # Dependent variable

# Function to create polynomial features for multiple features
def polynomial_features(X, degree):
    X_poly = np.ones((X.shape[0], 1))  # Bias term (intercept)
    for d in range(1, degree + 1):
        for i in range(X.shape[1]):  # Loop over each feature
            X_poly = np.hstack((X_poly, (X[:, i:i+1] ** d)))  # Adding polynomial terms
    return X_poly

# Set polynomial degree
degree = 2

# Transform X to polynomial features
X_poly = polynomial_features(X, degree)

# Standardize features (optional, improves numerical stability)
scaler = StandardScaler()
X_poly = scaler.fit_transform(X_poly)

# Compute theta using Normal Equation: θ = (X'X)^(-1) X'y
theta = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)

# Predict values
y_pred = X_poly.dot(theta)

# Print theta values
print("Optimized Theta values:")
print(theta)

# Predict the price for a new house with Area=2500 sqft, Bedrooms=3, Floors=2
new_house = np.array([[2500, 3, 2, 10]])  # Adding House Age (e.g., 10 years)
new_house_poly = polynomial_features(new_house, degree)  # Transform input
new_house_poly = scaler.transform(new_house_poly)  # Scale input
predicted_price = new_house_poly.dot(theta)  # Make prediction

print(f"Predicted price for the house: ${predicted_price[0][0]:,.2f}")
