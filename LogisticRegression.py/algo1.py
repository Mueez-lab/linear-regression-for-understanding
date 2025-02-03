import numpy as np
import matplotlib.pyplot as plt

# Sample Data: (Feature: Area, Target: Price category [0 or 1])
# X is the feature (e.g., Area), y is the target (binary labels, e.g., 0 or 1)
X = np.array([1500, 1800, 2300, 3000, 3500, 4000]).reshape(-1, 1)  # Area in sqft
y = np.array([0, 0, 0, 1, 1, 1])  # Binary labels for price (0 = low, 1 = high)

# Step 1: Initialize parameters
theta_0 = 0  # Bias term (intercept)
theta_1 = 0  # Weight (coefficient for the feature)
alpha = 0.01  # Learning rate
iterations = 5000  # Number of iterations
m = len(X)  # Number of data points

# Step 2: Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 3: Cost function (Log loss)
def cost_function(X, y, theta_0, theta_1):
    predictions = sigmoid(theta_0 + theta_1 * X)
    return (-1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

# Step 4: Gradient Descent
def gradient_descent(X, y, theta_0, theta_1, alpha, iterations):
    cost_history = []
    
    for _ in range(iterations):
        # Compute hypothesis (predictions)
        predictions = sigmoid(theta_0 + theta_1 * X)
        
        # Compute gradients
        grad_0 = (1/m) * np.sum(predictions - y)
        grad_1 = (1/m) * np.sum((predictions - y) * X)
        
        # Update the parameters
        theta_0 -= alpha * grad_0
        theta_1 -= alpha * grad_1
        
        # Record the cost to plot the cost history
        cost_history.append(cost_function(X, y, theta_0, theta_1))
    
    return theta_0, theta_1, cost_history

# Step 5: Train the model
theta_0, theta_1, cost_history = gradient_descent(X, y, theta_0, theta_1, alpha, iterations)

# Step 6: Make predictions
def predict(X, theta_0, theta_1):
    return sigmoid(theta_0 + theta_1 * X) >= 0.5  # Threshold at 0.5 to predict 0 or 1

# Step 7: Plot the cost history
plt.plot(range(iterations), cost_history, label="Cost function")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost function over iterations")
plt.legend()
plt.show()

# Step 8: Print the results
print("Optimized Parameters:")
print(f"0 (Bias): {theta_0}")
print(f"1 (Weight): {theta_1}")

# Predicting for a new value (e.g., Area = 2500 sqft)
new_area = 2500
prediction = predict(np.array([new_area]), theta_0, theta_1)
print(f"Predicted Price category for {new_area} sqft: {'High' if prediction[0] == 1 else 'Low'}")
