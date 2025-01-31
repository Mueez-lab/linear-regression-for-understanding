import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('C:/Users/mueez/OneDrive/Desktop/Machine/HousePrediction/file.csv');
print(df)

#Distribution of Data
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df.Area, df.Price, color='red', marker='x')
# plt.show()

reg = LinearRegression()
reg.fit(df[['Area']], df['Price'])  #df[['Area']] → Selects the Area column as a DataFrame (needed for Scikit-Learn). df['Price'] → Selects the Price column as the target (y).
y = reg.predict(pd.DataFrame([[3000]], columns=['Area']))
print(y)

print(reg.coef_) # shows the value of w aka m ie slope
print(reg.intercept_) # shows the value of b 

# mx + b 
print(135.78767123 * 3000 + 180616.43835616432)