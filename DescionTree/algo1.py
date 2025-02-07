import numpy as np
import pandas as pd
from collections import Counter

# Calculate Entropy
def entropy(y):
    class_counts = Counter(y)
    total = len(y)
    return -sum((count/total) * np.log2(count/total) for count in class_counts.values())

# Calculate Information Gain
def information_gain(data, target, feature):
    total_entropy = entropy(target)
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = sum((counts[i] / sum(counts)) * entropy(target[data[feature] == v]) for i, v in enumerate(values))
    return total_entropy - weighted_entropy

# ID3 Decision Tree Algorithm
def id3(data, target, features):
    if len(set(target)) == 1:
        return target.iloc[0]  # Return class if all instances have the same label
     
    if len(features) == 0:
        return target.mode()[0]  # Return majority class
    
    best_feature = max(features, key=lambda f: information_gain(data, target, f))
    
    tree = {best_feature: {}}
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        tree[best_feature][value] = id3(subset, subset[target.name], [f for f in features if f != best_feature])
    return tree

# Sample Dataset
data = pd.DataFrame({
    'Temp': ['Mild', 'Cool', 'Cool', 'Mild', 'Mild'],
    'Humidity': ['High', 'Normal', 'Normal', 'Normal', 'High'],
    'Wind': ['Weak', 'Weak', 'Strong', 'Weak', 'Strong'],
    'Play Tennis': ['Yes', 'Yes', 'No', 'Yes', 'No']
})

# Train Decision Tree
features = ['Temp', 'Humidity', 'Wind']
target = data['Play Tennis']
decision_tree = id3(data, target, features)

# Print Tree
import pprint
pprint.pprint(decision_tree)