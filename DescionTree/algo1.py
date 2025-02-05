import numpy as np
import pandas as pd
from collections import Counter

# Function to calculate entropy
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# Function to split dataset
def split_dataset(X, y, feature, threshold):
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

# Function to find the best split
def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None
    current_entropy = entropy(y)
    
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            p_left = len(y_left) / len(y)
            p_right = len(y_right) / len(y)
            
            gain = current_entropy - (p_left * entropy(y_left) + p_right * entropy(y_right))
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold

# Class for Decision Tree
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y, depth=0):
        """Builds the decision tree and stores it in self.tree"""
        self.tree = self._build_tree(X, y, depth)
    
    def _build_tree(self, X, y, depth):
        """Recursive function to build the tree"""
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0]
        
        feature, threshold = best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        
        X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)
        
        node = {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1)
        }
        return node
    
    def predict_sample(self, node, sample):
        if not isinstance(node, dict):
            return node
        if sample[node['feature']] <= node['threshold']:
            return self.predict_sample(node['left'], sample)
        else:
            return self.predict_sample(node['right'], sample)
    
    def predict(self, X):
        """Predict labels for given dataset"""
        return np.array([self.predict_sample(self.tree, sample) for sample in X])

# Sample dataset
X = np.array([[2.7], [1.3], [3.5], [0.8], [2.2], [3.0], [1.0], [2.5], [3.8], [0.5]])
y = np.array([1, 0, 1, 0, 1, 1, 0, 1, 1, 0])

# Create and train decision tree
dt = DecisionTree(max_depth=3)
dt.fit(X, y)

# Test prediction
X_test = np.array([[2.0], [3.0], [1.5], [3.7]])
predictions = dt.predict(X_test)
print("Predictions:", predictions)
