# Step 1: Import required libraries

import numpy as np                 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error   


# Step 2: Generate sample (synthetic) data

np.random.seed(42)  
x = np.linspace(0, 10, 100).reshape(-1, 1)


# Define target values y using the equation y = 2x + noise
y = 2 * x.flatten() + np.random.normal(0, 1, 100)


print("\nMatrix X (input features):")
print(x[:10]) 

print("\nVector y (target values):")
print(y[:10])


# Step 3: Train the Linear Regression model

model = LinearRegression()

model.fit(x, y)

slope = model.coef_[0]
intercept = model.intercept_

# Print the model equation
print(f"Trained model: y = {slope:.2f}x + {intercept:.2f}")


# Step 4: Make predictions and evaluate performance

y_pred = model.predict(x)

mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.2f}")


new_x = np.array([[5]]) 
new_y_pred = model.predict(new_x)
print(f"Prediction for x=5: {new_y_pred[0]:.2f}")


# Step 5: Visualize results

plt.figure(figsize=(8, 5))      

plt.scatter(x, y, color='blue', label='Data points')

plt.plot(x, 2 * x, color='green', linestyle='--', label='True line (y=2x)')

plt.plot(x, y_pred, color='red', label='Predicted line')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression with Scikit-Learn')
plt.legend()
plt.show()