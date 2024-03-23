import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to load and preprocess the data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    X = df[['AT', 'V', 'AP', 'RH']]  # Features
    y = df['PE']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return X_train_normalized, X_test_normalized, y_train, y_test

# Function to train Linear Regression model
def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_train), model.predict(X_test)

# Function to train Random Forest model
def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_train), model.predict(X_test)

# Function to train Neural Network model
def train_neural_network(X_train, X_test, y_train, y_test):
    model = MLPRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_train), model.predict(X_test)

# Function to evaluate model and print metrics
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print("R2:", r2)
    print("MAE:", mae)
    print("MAPE:", mape)
    print("MSE:", mse)

# Function to visualize actual vs predicted values
def visualize_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', label='Actual vs. Predicted')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], linestyle='--', color='red', label='Perfect Prediction')
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load and preprocess the data
X_train, X_test, y_train, y_test = load_and_preprocess_data('c2p2.csv')

# Train Linear Regression model
lr_train_pred, lr_test_pred = train_linear_regression(X_train, X_test, y_train, y_test)
print("Linear Regression Model Evaluation:")
evaluate_model(y_train, lr_train_pred)
evaluate_model(y_test, lr_test_pred)
visualize_actual_vs_predicted(y_test, lr_test_pred, "Actual vs Predicted (Linear Regression)")

# Train Random Forest model
rf_train_pred, rf_test_pred = train_random_forest(X_train, X_test, y_train, y_test)
print("\nRandom Forest Model Evaluation:")
evaluate_model(y_train, rf_train_pred)
evaluate_model(y_test, rf_test_pred)
visualize_actual_vs_predicted(y_test, rf_test_pred, "Actual vs Predicted (Random Forest)")

# Train Neural Network model
nn_train_pred, nn_test_pred = train_neural_network(X_train, X_test, y_train, y_test)
print("\nNeural Network Model Evaluation:")
evaluate_model(y_train, nn_train_pred)
evaluate_model(y_test, nn_test_pred)
visualize_actual_vs_predicted(y_test, nn_test_pred, "Actual vs Predicted (Neural Network)")

# Compare models and select the best one based on R2 score
models = ['Linear Regression', 'Random Forest', 'Neural Network']
r2_scores = [r2_score(y_test, lr_test_pred), r2_score(y_test, rf_test_pred), r2_score(y_test, nn_test_pred)]
best_model_index = np.argmax(r2_scores)
print("\nBest Model:", models[best_model_index])
