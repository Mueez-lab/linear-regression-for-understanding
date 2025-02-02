import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('C:/Users/mueez/OneDrive/Desktop/Machine/MultiLinearRegression/file.csv')

# Extract features and target variable
X = data[['Size (sqft)', 'Bedrooms', 'Bathrooms', 'House Age (years)']].values # 2d
y = data['Price'].values.reshape(-1, 1)  # Reshaping y to be a column vector  1d converted into 2d array # -1 tells NumPy to automatically infer the correct number of rows based on the available data. and 1 tells that 1 column 

# Feature Normalization
def feature_normalize(X):
    """Performs feature scaling: Mean Normalization"""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# Add intercept term (column of ones) to X
def add_intercept(X):
    m = X.shape[0]
    X_with_intercept = np.c_[np.ones((m, 1)), X]  # Add a column of ones
    return X_with_intercept

# Compute cost function J(θ)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost

# Gradient Descent to optimize θ
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    
    for _ in range(num_iters):
        gradient = (1 / m) * (X.T.dot(X.dot(theta) - y))
        theta -= alpha * gradient
        J_history.append(compute_cost(X, y, theta))
    
    return theta, J_history

# Normalize features
X_norm, mu, sigma = feature_normalize(X)

# Add intercept term to X
X_norm = add_intercept(X_norm)

# Initialize theta to zeros
theta = np.zeros((X_norm.shape[1], 1))

# Set learning rate and number of iterations
alpha = 0.01
num_iters = 1000

# Run gradient descent to learn theta
theta, J_history = gradient_descent(X_norm, y, theta, alpha, num_iters)

# Output the optimized theta values and the cost function history
print("Optimized Theta values:")
print(theta)
print("\nCost function history:")
print(J_history[:-1])  # Display the final cost after training

# Predict the price for a new house
new_house = np.array([1, (2500 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1], (2 - mu[2]) / sigma[2], (10 - mu[3]) / sigma[3]])  # Normalized new feature values
predicted_price = new_house.dot(theta)
print("\nPredicted price for a new house: $", predicted_price[0])