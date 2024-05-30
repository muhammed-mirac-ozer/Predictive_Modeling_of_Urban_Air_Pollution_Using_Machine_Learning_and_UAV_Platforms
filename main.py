import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.dates as mdates
import time
import pandas as pd

start_time = time.time()

# Load your dataset
data = pd.read_excel("sample_data.xlsx")

# Replace ',' with '.' and convert to numeric for relevant columns
data[['PM10', 'PM 2.5', 'SO2', 'CO', 'NO2', 'NOX', 'NO', 'O3']] = data[['PM10', 'PM 2.5', 'SO2', 'CO', 'NO2', 'NOX', 'NO', 'O3']].replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce')

# Fill missing values with the nearest non-null value
data_filled = data.fillna(method='ffill')

# Save the filled data to a new Excel file
data_filled.to_excel("sample_data_filled.xlsx", index=False)

# Separate features and target variable
X = data_filled.drop(['Date', 'O3'], axis=1)
y = data_filled['O3']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preprocessing pipeline
model_pipeline = make_pipeline(
    StandardScaler()
)

# Fit and transform features
x_train_preprocessed = model_pipeline.fit_transform(x_train)
x_test_preprocessed = model_pipeline.transform(x_test)


# Load dataset
data = pd.read_excel("sample_data.xlsx")

# Calculate correlation on numeric columns in data set
correlation_matrix = data.corr()

# View correlation matrix
print(correlation_matrix)

# Linear Regression
lr_model = LinearRegression().fit(x_train_preprocessed, y_train)
y_pred_lr = lr_model.predict(x_test_preprocessed)

# Decision Tree
dtr_params = {"max_depth": [5, 8, 10],
              "min_samples_split": [2, 10, 80, 100],
              "min_samples_leaf": [10, 15, 20]}
dtr_grid = GridSearchCV(DecisionTreeRegressor(), dtr_params, cv=5)
dtr_grid.fit(x_train, y_train)
dtr_best_params = dtr_grid.best_params_
dtr_model = DecisionTreeRegressor(**dtr_best_params).fit(x_train, y_train)
y_pred_dtr = dtr_model.predict(x_test)

# Random Forest
rf_params = {"max_depth": [5, 8, 10],
              "min_samples_split": [2, 10, 80, 100],
              "min_samples_leaf": [10, 15, 20]}
rf_grid = GridSearchCV(RandomForestRegressor(), rf_params, cv=5)
rf_grid.fit(x_train, y_train)
rf_best_params = rf_grid.best_params_
rf_model = RandomForestRegressor(**rf_best_params).fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)

# Support Vector Regression
svr_params = {'C': [0.1, 1, 10, 100, 1000]}
svr_grid = GridSearchCV(SVR(), svr_params, cv=5)
svr_grid.fit(x_train_preprocessed, y_train)
svr_best_params = svr_grid.best_params_
sv_model = SVR(**svr_best_params).fit(x_train_preprocessed, y_train)
y_pred_sv = sv_model.predict(x_test_preprocessed)

# k-Nearest Neighbors
kn_model = KNeighborsRegressor(n_neighbors=10).fit(x_train, y_train)
y_pred_kn = kn_model.predict(x_test)

# Multi-layer Perceptron
mlp_params = {"alpha": [0.1, 0.01, 0.02, 0.001, 0.0001],
              "hidden_layer_sizes": [(10, 20), (5, 5), (100, 100)]}
mlp_grid = GridSearchCV(MLPRegressor(), mlp_params, cv=5)
mlp_grid.fit(x_train, y_train)
mlp_best_params = mlp_grid.best_params_
mlp_model = MLPRegressor(**mlp_best_params).fit(x_train, y_train)
y_pred_mlp = mlp_model.predict(x_test)

# Comparison
models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Support Vector Regression', 'k-Nearest Neighbors', 'Multi-layer Perceptron']
results = []

for idx, y_pred in enumerate([y_pred_lr, y_pred_dtr, y_pred_rf, y_pred_sv, y_pred_kn, y_pred_mlp]):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    results.append((models[idx], rmse, r2, mae, mse, mape))

results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'R2', 'MAE', 'MSE', 'MAPE'])
print(results_df)

# Convert date column to datetime format
date_filled['Date'] = pd.to_datetime(date_filled['Date'])

# select appropriate date data for x-axis
x_test_dates = date_filled['Date'].iloc[x_test.index]

# Sort data
sorted_indices = x_test_dates.argsort()
x_test_dates = x_test_dates.iloc[sorted_indices]
y_test_sorted = y_test.iloc[sorted_indices]

# Visualization
colors = ['blue', 'green', 'red', 'purple', 'orange']

for idx, y_pred in enumerate([y_pred_lr, y_pred_dtr, y_pred_rf, y_pred_sv, y_pred_kn, y_pred_mlp]):
    plt.figure(figsize=(12, 6))
    plt.plot(x_test_dates, y_test_sorted.values, label='Actual', alpha=0.5, color='black')  # Keeps Actual in black
    plt.plot(x_test_dates, y_pred[sorted_indices], label=models[idx], color=colors[idx])
    plt.title(models[idx], fontdict={'fontname': 'Times New Roman', 'fontweight': 'bold'})
    plt.xlabel('Date and Time', fontname='Times New Roman')
    plt.ylabel('O3 Value', fontname='Times New Roman')
    plt.legend()

    # Set x-axis boundaries (4.08.2021 - 17.12.2022)
    plt.xlim(pd.to_datetime('2021-08-04'), pd.to_datetime('2022-12-17'))

    # Adjustments for bi-weekly tag display
    locator = mdates.DayLocator(interval=14)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # To show dates more frequently
    plt.xticks(rotation=45, ha='right')  # Etiketlerin sağa hizalanması
    plt.show()

# Visualization
colors = ['blue', 'green', 'red', 'purple', 'orange']

for idx, (y_pred, model_name) in enumerate(
        zip([y_pred_lr, y_pred_dtr, y_pred_rf, y_pred_sv, y_pred_kn, y_pred_mlp], models)):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.values, y_pred, 'o', label='Measured vs Predicted', color=colors[idx])
    ax.plot(y_test.values, y_test.values, label='Measure', linestyle='--', color='black')  # Add line for measure
    ax.set_title(f'{model_name} - Measured vs Predicted', fontdict={'fontname': 'Times New Roman', 'fontweight': 'bold'})
    ax.set_xlabel('Measured O3', fontname='Times New Roman')
    ax.set_ylabel('Predicted O3', fontname='Times New Roman')
    ax.legend()
    # set x-axis boundaries
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 110)
    plt.show()

end_time = time.time()
total_time = end_time - start_time

print(f"Code execution timei: {total_time} second")

# Calculate standard deviation values
std_lr = np.std(y_pred_lr)
std_dtr = np.std(y_pred_dtr)
std_rf = np.std(y_pred_rf)
std_sv = np.std(y_pred_sv)
std_kn = np.std(y_pred_kn)
std_mlp = np.std(y_pred_mlp)

# Store standard deviation values as a list
std_values = [std_lr, std_dtr, std_rf, std_sv, std_kn, std_mlp]

# Add standard deviation column to Table.
results_df['σ'] = std_values

# Update your table
print(results_df)

# prepared by Muhammed Mirac Ozer.