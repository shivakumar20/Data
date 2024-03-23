# %%


import pandas as pd

df = pd.read_csv('c2p2.csv')

# %%
# Display basic information about the DataFrame
print("Basic Information about the DataFrame:")
print(df.info())

# Display the shape of the DataFrame (number of rows, number of columns)
print("\nShape of the DataFrame:")
print(df.shape)

# Display descriptive statistics for numerical columns
print("\nDescriptive Statistics:")
print(df.describe())

# Display the first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:")
print(df.head())

# Display the last few rows of the DataFrame
print("\nLast few rows of the DataFrame:")
print(df.tail())

# Display column names of the DataFrame
print("\nColumn Names:")
print(df.columns)

# Display data types of each column
print("\nData Types:")
print(df.dtypes)

# Check for missing values in the DataFrame
print("\nMissing Values:")
print(df.isnull().sum())



# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure and axis grid
fig, axes = plt.subplots(nrows=len(df.columns), ncols=2, figsize=(12, 8))
fig.tight_layout(pad=3.0)

# Loop through each column in the DataFrame
for i, column in enumerate(df.columns):
    # Plot histogram
    sns.histplot(data=df, x=column, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f"Histogram of {column}")
    axes[i, 0].set_xlabel(column)
    axes[i, 0].set_ylabel("Frequency")
    
    # Plot density plot
    sns.kdeplot(data=df, x=column, ax=axes[i, 1], fill=True)
    axes[i, 1].set_title(f"Density Plot of {column}")
    axes[i, 1].set_xlabel(column)
    axes[i, 1].set_ylabel("Density")

# Show the plots
plt.show()

# %%


# Assume 'dataframe' is your DataFrame loaded with data

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


# %%
# Pairplot for scatter visualization
sns.pairplot(df)
plt.show()

# %%
!pip install scikit-learn

# %%
from sklearn.model_selection import train_test_split

# Assume 'dataframe' is your DataFrame loaded with data
# Assume 'features' are columns AT, V, AP, RH, and 'target' is column PE

# Separate features and target variable
X = df[['AT', 'V', 'AP', 'RH']]  # Features
y = df['PE']  # Target variable

# Split the data into training and test sets at a 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Show the shape of the training and test sets
print("Shape of training features:", X_train.shape)
print("Shape of training target variable:", y_train.shape)
print("Shape of test features:", X_test.shape)
print("Shape of test target variable:", y_test.shape)

# %%
from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler instance
scaler = MinMaxScaler()

# Fit the scaler on the training features and transform them
X_train_normalized = scaler.fit_transform(X_train)

# Transform the test features using the scaler fitted on the training data
X_test_normalized = scaler.transform(X_test)

# Show the shape of the normalized training and test sets
print("Shape of normalized training features:", X_train_normalized.shape)
print("Shape of normalized test features:", X_test_normalized.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a linear regression model
model = LinearRegression()

# Train the model on the training dataset
model.fit(X_train_normalized, y_train)

# Predict the target variable for the training and testing data
y_train_pred = model.predict(X_train_normalized) 
y_test_pred = model.predict(X_test_normalized)

# Calculate the mean squared error for training and testing data
mse_train = mean_squared_error(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred)

# Display the mean squared error 
print("Mean Squared Error (Training):", mse_train) 
print("Mean Squared Error (Testing):", mse_test)

# %%
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Make predictions on training and test data
y_train_pred = model.predict(X_train_normalized)
y_test_pred = model.predict(X_test_normalized)

# Calculate R-squared (R2) for training and test data
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate Mean Absolute Error (MAE) for training and test data
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

# Calculate Mean Squared Error (MSE) for training and test data
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Calculate Root Mean Squared Error (RMSE) for training and test data
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# Print the evaluation metrics
print("Evaluation Metrics:")
print("R-squared (R2) - Training:", r2_train)
print("R-squared (R2) - Test:", r2_test)
print("Mean Absolute Error (MAE) - Training:", mae_train)
print("Mean Absolute Error (MAE) - Test:", mae_test)
print("Mean Squared Error (MSE) - Training:", mse_train)
print("Mean Squared Error (MSE) - Test:", mse_test)
print("Root Mean Squared Error (RMSE) - Training:", rmse_train)
print("Root Mean Squared Error (RMSE) - Test:", rmse_test)


# %%
import matplotlib.pyplot as plt

# Plot actual vs predicted values for the training data

plt.figure(figsize=(10, 6))

plt.scatter(y_train, y_train_pred, color='blue', label='Actual vs Predicted (Training)')

plt.xlabel('Actual PE')

plt.ylabel('Predicted PE')

plt.title('Actual vs Predicted Values (Training)')

plt.legend()

plt.show()

# Plot actual vs predicted values for the testing data

plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_test_pred, color='red', label='Actual vs Predicted (Testing)')

plt.xlabel('Actual PE')

plt.ylabel('Predicted PE')

plt.title('Actual vs Predicted Values (Testing)')

plt.legend()

plt.show()


# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Create a Random Forest regression model
random_forest_model = RandomForestRegressor(random_state=42)

# Train the model on the training dataset
random_forest_model.fit(X_train_normalized, y_train)

# Make predictions on training and test data
y_train_pred_rf = random_forest_model.predict(X_train_normalized)
y_test_pred_rf = random_forest_model.predict(X_test_normalized)

# Calculate evaluation metrics - MAE, MAPE, MSE, R2
mae_train_rf = mean_absolute_error(y_train, y_train_pred_rf)
mae_test_rf = mean_absolute_error(y_test, y_test_pred_rf)

mse_train_rf = mean_squared_error(y_train, y_train_pred_rf)
mse_test_rf = mean_squared_error(y_test, y_test_pred_rf)

r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)

# Calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_train_rf = mean_absolute_percentage_error(y_train, y_train_pred_rf)
mape_test_rf = mean_absolute_percentage_error(y_test, y_test_pred_rf)

# Print evaluation metrics
print("Random Forest Regression Model Evaluation:")
print("Training MAE:", mae_train_rf)
print("Testing MAE:", mae_test_rf)
print("Training MAPE:", mape_train_rf)
print("Testing MAPE:", mape_test_rf)
print("Training MSE:", mse_train_rf)
print("Testing MSE:", mse_test_rf)
print("Training R2:", r2_train_rf)
print("Testing R2:", r2_test_rf)

# Visualize actual vs. predicted values for PE
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred_rf, color='blue', label='Actual vs. Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red', label='Perfect Prediction')
plt.title('Actual vs. Predicted Values for PE (Random Forest)')
plt.xlabel('Actual PE')
plt.ylabel('Predicted PE')
plt.legend()
plt.grid(True)
plt.show()


# %%
import matplotlib.pyplot as plt

# Plot actual vs predicted values for both training and testing data

plt.figure(figsize=(10, 6))

plt.scatter(y_train, y_train_pred_rf, color='blue', label='Actual vs Predicted (Training)')
plt.scatter(y_test, y_test_pred_rf, color='red', label='Actual vs Predicted (Testing)')

plt.xlabel('Actual PE')
plt.ylabel('Predicted PE')

plt.title('Actual vs Predicted Values (Random Forest)')
plt.legend()

plt.show()


# %%
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Create a Multi-layer Perceptron regressor model
mlp_model = MLPRegressor(random_state=42)

# Train the model on the training dataset
mlp_model.fit(X_train_normalized, y_train)

# Make predictions on training and test data
y_train_pred_mlp = mlp_model.predict(X_train_normalized)
y_test_pred_mlp = mlp_model.predict(X_test_normalized)

# Calculate evaluation metrics - MAE, MAPE, MSE, R2
mae_train_mlp = mean_absolute_error(y_train, y_train_pred_mlp)
mae_test_mlp = mean_absolute_error(y_test, y_test_pred_mlp)

mse_train_mlp = mean_squared_error(y_train, y_train_pred_mlp)
mse_test_mlp = mean_squared_error(y_test, y_test_pred_mlp)

r2_train_mlp = r2_score(y_train, y_train_pred_mlp)
r2_test_mlp = r2_score(y_test, y_test_pred_mlp)

# Calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_train_mlp = mean_absolute_percentage_error(y_train, y_train_pred_mlp)
mape_test_mlp = mean_absolute_percentage_error(y_test, y_test_pred_mlp)

# Print evaluation metrics
print("Neural Network Regression Model Evaluation:")
print("Training MAE:", mae_train_mlp)
print("Testing MAE:", mae_test_mlp)
print("Training MAPE:", mape_train_mlp)
print("Testing MAPE:", mape_test_mlp)
print("Training MSE:", mse_train_mlp)
print("Testing MSE:", mse_test_mlp)
print("Training R2:", r2_train_mlp)
print("Testing R2:", r2_test_mlp)

# Visualize actual vs. predicted values for PE
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred_mlp, color='blue', label='Actual vs. Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red', label='Perfect Prediction')
plt.title('Actual vs. Predicted Values for PE (Neural Network)')
plt.xlabel('Actual PE')
plt.ylabel('Predicted PE')
plt.legend()
plt.grid(True)
plt.show()


# %%
import matplotlib.pyplot as plt

# Create a scatter plot for actual vs. predicted values for both training and testing data
plt.figure(figsize=(10, 6))

# Plot actual vs. predicted values for the training data
plt.scatter(y_train, y_train_pred_mlp, color='blue', label='Actual vs. Predicted (Training)', alpha=0.5)

# Plot actual vs. predicted values for the testing data
plt.scatter(y_test, y_test_pred_mlp, color='red', label='Actual vs. Predicted (Testing)', alpha=0.5)

# Add labels and title
plt.xlabel('Actual PE')
plt.ylabel('Predicted PE')
plt.title('Actual vs. Predicted Values for PE (Neural Network)')

# Add legend
plt.legend()

# Add grid
plt.grid(True)

# Show plot
plt.show()


# %%
# Calculate Mean Absolute Percentage Error (MAPE) for Random Forest
mape_test_rf = mean_absolute_percentage_error(y_test, y_test_pred_rf)

# Calculate Mean Absolute Percentage Error (MAPE) for Linear Regression
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)



# %%
import pandas as pd

# Create a list to store evaluation metrics for LR, RF, and NN
evaluation_data = []

# Add evaluation metrics for Linear Regression (LR) model
evaluation_data.append({'Model': 'Linear Regression (LR)',
                         'R2': r2_test,
                         'MAE': mae_test,
                         'MAPE': mape_test,
                         'MSE': mse_test})

# Add evaluation metrics for Random Forest (RF) model
evaluation_data.append({'Model': 'Random Forest (RF)',
                         'R2': r2_test_rf,
                         'MAE': mae_test_rf,
                         'MAPE': mape_test_rf,
                         'MSE': mse_test_rf})

# Add evaluation metrics for Neural Network (NN) model
evaluation_data.append({'Model': 'Neural Network (NN)',
                         'R2': r2_test_mlp,
                         'MAE': mae_test_mlp,
                         'MAPE': mape_test_mlp,
                         'MSE': mse_test_mlp})

# Create DataFrame from the list of evaluation metrics
evaluation_df = pd.DataFrame(evaluation_data)

# Display the DataFrame
print("Evaluation Metrics for Each Model:")
print(evaluation_df)

# Select the best model based on evaluation metrics (e.g., highest R2, lowest MAE, lowest MSE)
best_model = evaluation_df.loc[evaluation_df['R2'].idxmax()]

print("\nBest Model:")
print(best_model)


# %% [markdown]
# C2 P2 FORCASTING

# %%



