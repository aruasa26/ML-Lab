import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('Nairobi_office_price.csv')
x = data['SIZE'].values
y = data['PRICE'].values


# Define MSE function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Define Gradient Descent
def gradient_descent(x, y, m, c, learning_rate, epochs):
    N = len(y)
    for epoch in range(epochs):
        y_pred = m * x + c
        error = mean_squared_error(y, y_pred)
        print(f"Epoch {epoch + 1}/{epochs}, MSE: {error}")

        m_gradient = (-2 / N) * np.sum(x * (y - y_pred))
        c_gradient = (-2 / N) * np.sum(y - y_pred)

        m -= learning_rate * m_gradient
        c -= learning_rate * c_gradient

    return m, c


# Initialize parameters
m_initial = np.random.rand()
c_initial = np.random.rand()
learning_rate = 0.0001
epochs = 10

# Train model
m_trained, c_trained = gradient_descent(x, y, m_initial, c_initial, learning_rate, epochs)
print(f"Trained slope (m): {m_trained}, Trained intercept (c): {c_trained}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, m_trained * x + c_trained, color='red', label='Line of Best Fit')
plt.xlabel("Office Size (sq. ft)")
plt.ylabel("Office Price")
plt.title("Office Size vs. Price with Line of Best Fit")
plt.legend()
plt.show()

# Predict for 100 sq. ft
size = 100
predicted_price = m_trained * size + c_trained
print(f"Predicted price for 100 sq. ft office: {predicted_price}")
