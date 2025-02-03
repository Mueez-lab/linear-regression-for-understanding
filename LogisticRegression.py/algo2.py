import numpy as np
import matplotlib.pyplot as plt

# Sample dataset with multiple features (e.g., Area, Bedrooms)
X = np.array([
    [1500, 3], 
    [1800, 3], 
    [2300, 4], 
    [3000, 4], 
    [3500, 5], 
    [4000, 5]
])  # Features: [Area, Bedrooms]

y = np.array([0, 0, 0, 1, 1, 1])  # Labels (0 = low price, 1 = high price)

# Feature Scaling: Standardize each feature
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Add bias term (column of ones) to X
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Shape becomes (m, n+1)

# Initialize parameters
theta = np.zeros(X.shape[1])  # Initialize all weights to zero
alpha = 0.01  # Learning rate
iterations = 5000  # Number of iterations
m = len(y)  # Number of samples

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (Log loss)
def cost_function(X, y, theta):
    predictions = sigmoid(X @ theta)  # Matrix multiplication
    return (-1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

# Gradient Descent (Vectorized)
def gradient_descent(X, y, theta, alpha, iterations):
    cost_history = []
    
    for _ in range(iterations):
        predictions = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (predictions - y))  # Vectorized gradient calculation
        theta -= alpha * gradient  # Update parameters
        cost_history.append(cost_function(X, y, theta))  # Store cost for plotting
    
    return theta, cost_history

# Train the model
theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

# Prediction function
def predict(X, theta):
    return sigmoid(X @ theta) >= 0.5  # Apply threshold at 0.5

# Plot Cost History
plt.plot(range(iterations), cost_history, label="Cost function")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost function over iterations")
plt.legend()
plt.show()

# Print optimized parameters
print("Optimized Parameters:")
print(theta)

# Predicting for a new house (Area = 2500 sqft, 4 bedrooms)
new_house = np.array([2500, 4])
new_house_scaled = (new_house - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)  # Scale features
new_house_scaled = np.hstack(([1], new_house_scaled))  # Add bias term

prediction = predict(new_house_scaled, theta)
print(f"Predicted Price category for a 2500 sqft house with 4 bedrooms: {'High' if prediction else 'Low'}")
