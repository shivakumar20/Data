import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Load the data
df = pd.read_csv('c2p2.csv')

# Function to preprocess the data
def preprocess_data(df):
    scaler = MinMaxScaler()
    X = df[['AT', 'V', 'AP', 'RH']].values
    X_normalized = scaler.fit_transform(X)
    return X_normalized

# Function to make predictions using Linear Regression
def predict_linear_regression(data):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(data)

# Function to make predictions using Random Forest
def predict_random_forest(data):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model.predict(data)

# Function to make predictions using Neural Network
def predict_neural_network(data):
    model = MLPRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model.predict(data)

# Function to handle prediction button click
def predict():
    data = np.array([[float(entry.get()) for entry in entries]])
    model_name = selected_model.get()
    if model_name == 'Linear Regression':
        prediction = predict_linear_regression(data)
    elif model_name == 'Random Forest':
        prediction = predict_random_forest(data)
    elif model_name == 'Neural Network':
        prediction = predict_neural_network(data)
    else:
        prediction = "Select a model"
    
    # Extract the prediction value from the array
    if isinstance(prediction, np.ndarray):
        prediction_value = prediction[0]
    else:
        prediction_value = prediction
    
    # Update the result label with the prediction
    result_label.config(text=f"Predicted PE: {prediction_value:.2f}")


# Create the main application window
root = tk.Tk()
root.title("Power Plant Prediction")

# Preprocess the data
X_train = preprocess_data(df[['AT', 'V', 'AP', 'RH']])
y_train = df['PE'].values.reshape(-1, 1)

# Dropdown to select model
model_label = tk.Label(root, text="Select Model:")
model_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

selected_model = tk.StringVar()
model_dropdown = ttk.Combobox(root, textvariable=selected_model, state="readonly")
model_dropdown['values'] = ('Linear Regression', 'Random Forest', 'Neural Network')
model_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
model_dropdown.current(0)

# Entry fields for input data
inputs_label = tk.Label(root, text="Input Data:")
inputs_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

entries = []
for i, feature in enumerate(['AT', 'V', 'AP', 'RH']):
    entry_label = tk.Label(root, text=feature)
    entry_label.grid(row=i+2, column=0, padx=5, pady=5, sticky="w")
    entry = tk.Entry(root)
    entry.grid(row=i+2, column=1, padx=5, pady=5, sticky="w")
    entries.append(entry)

# Button to predict
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=6, column=0, columnspan=2, padx=5, pady=10)

# Label to display result
result_label = tk.Label(root, text="")
result_label.grid(row=7, column=0, columnspan=2, padx=5, pady=5)

# Run the main event loop
root.mainloop()
