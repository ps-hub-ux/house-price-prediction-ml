from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = data.target

# Split data
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Loading dataset...")
data = fetch_california_housing()

print("Creating dataframe...")
df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = data.target

print("Splitting data...")
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")
from sklearn.metrics import mean_squared_error

# Predict using test data
predictions = model.predict(X_test)

# Compare actual vs predicted
print("Actual values:", y_test[:5].values)
print("Predicted values:", predictions[:5])

# Calculate error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
import matplotlib.pyplot as plt

plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()