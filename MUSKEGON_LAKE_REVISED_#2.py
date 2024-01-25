#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Really need these
import pandas as pd 
import numpy as np
from numpy import *
from scipy.stats import skew
from statsmodels.tsa.seasonal import STL

# Handy for debugging
import gc
import time
import warnings
import os

# Date stuff
from datetime import datetime
from datetime import timedelta

# Nice graphing tools
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.offline as py
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
from matplotlib.dates import MonthLocator, YearLocator, DateFormatter
import seaborn as sns

# Machine learning tools
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Performance measures
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
###################################################################################################################
# Load the data from the csv file,dont forget the extra backslahses
Path = "C:\\Users\\cbeau\\Downloads\\MUSKEGON_LAKE\\MUSKEGON_LAKE_ DATA.xlsx"
ML= pd.read_excel(Path)

# Ensure you have an up-to-date version of pandas
print(pd.__version__)

# Convert date columns to datetime
date_columns = ['x_date']  # Replace with your date column names
for date_column in date_columns:
    ML[date_column] = pd.to_datetime(ML[date_column])

# Convert other columns to numeric
for column in ML.columns:
    if column not in date_columns:
        ML[column] = pd.to_numeric(ML[column], errors='coerce')
################################################################################################################
#Time series of DO 2m,5m,8m,11m
#Threholds from “Temporal Prediction of Coastal Water Quality Based on Environmental Factors with Machine Learning”

ML.set_index('x_date', inplace=True)


# Calculating monthly averages for the filtered and interpolated data
monthly_avg = ML.resample('M').mean()

# Define the hypoxia thresholds
mild_hypoxia_threshold = 4  # mg/L
severe_hypoxia_threshold = 2  # mg/L

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(monthly_avg['do2m'], label='DO 2m', color='blue')
plt.plot(monthly_avg['do5m'], label='DO 5m', color='green')
plt.plot(monthly_avg['do8m'], label='DO 8m', color='red')
plt.plot(monthly_avg['do11m'], label='DO 11m', color='purple')

# Hypoxia thresholds
plt.axhline(y=mild_hypoxia_threshold, color='orange', linestyle='--', label='Mild Hypoxia Threshold')
plt.axhline(y=severe_hypoxia_threshold, color='brown', linestyle='--', label='Severe Hypoxia Threshold')

# Adding titles and labels
plt.title('Monthly Average of Dissolved Oxygen')
plt.xlabel('Date')
plt.ylabel('Dissolved Oxygen (mg/L)')
plt.legend()

# Display the plot
plt.show()
#########################################################################################

# Select the columns for the new dataset
selected_columns = [
    'wt2m', 'wt4m', 'wt6m', 'wt7m', 'wt9m', 'wt11m', 
    'wspd', 'wdir1', 'do8m', 'do11m'
]

# Create the new dataset
ml = ML[selected_columns].copy()

# Filter data to include only between 5/1 and 11/1 each year
# and exclude data after 11/1/2022
ml = ml[(ml.index.month >= 5) & (ml.index.month <= 11) & (ml.index < '2022-11-02')]

# Display the first few rows of the modified dataset
print(ml.head())
################################################################################################################
#Evident is the disocnitunity in the data
# Make the time series continuous
# Set frequency to daily and reindex to fill missing dates with NaNs
ml = ml.asfreq('D')
ml = ml.reindex(pd.date_range(start=ml.index.min(), end=ml.index.max(), freq='D'))

# Resample the data to get weekly averages
# 'W' stands for weekly frequency
ml_weekly = ml.resample('W').mean()

# Create an instance of KNNImputer
# n_neighbors can be adjusted based on your dataset
imputer = KNNImputer(n_neighbors=5, weights='uniform')

# Apply KNN imputation
ml_weekly_imputed = imputer.fit_transform(ml_weekly)

# Convert the imputed data back to a DataFrame
ml_weekly_imputed_df = pd.DataFrame(ml_weekly_imputed, columns=ml_weekly.columns, index=ml_weekly.index)

# Display the first few rows of the imputed dataset
print(ml_weekly_imputed_df.head())

# Check for remaining NaNs
nan_counts_after_imputation = ml_weekly_imputed_df.isna().sum()
print("Number of NaN values in each column after KNN imputation:\n", nan_counts_after_imputation)
##########################################################################################################
# Features (excluding 'do8m' and 'do11m')
feature_columns = [col for col in ml_weekly_imputed_df.columns if col not in ['do8m', 'do11m']]
X = ml_weekly_imputed_df[feature_columns]

# Target variables
y_do8m = ml_weekly_imputed_df['do8m']
y_do11m = ml_weekly_imputed_df['do11m']
################################################################################################
#70% of the data for training and 30% for testing
# Calculate the index to split the data
split_index = int(len(ml_weekly_imputed_df) * 0.7)

# Temporal split for the features
X_train = ml_weekly_imputed_df[feature_columns][:split_index]
X_test = ml_weekly_imputed_df[feature_columns][split_index:]

# Temporal split for 'do8m'
y_train_do8m = ml_weekly_imputed_df['do8m'][:split_index]
y_test_do8m = ml_weekly_imputed_df['do8m'][split_index:]
dates_8m = y_test_do8m.index  # Test dates for 'do8m'

# Temporal split for 'do11m'
y_train_do11m = ml_weekly_imputed_df['do11m'][:split_index]
y_test_do11m = ml_weekly_imputed_df['do11m'][split_index:]
dates_11m = y_test_do11m.index  # Test dates for 'do11m'
#######################################################################################################
#Training the Random Forest Models
# Train the model for 'do8m'
rf_do8m = RandomForestRegressor(n_estimators=100, random_state=42)
rf_do8m.fit(X_train, y_train_do8m)

# Train the model for 'do11m'
rf_do11m = RandomForestRegressor(n_estimators=100, random_state=42)
rf_do11m.fit(X_train, y_train_do11m)
#################################################################################################
# Make predictions for 'do8m'
y_pred_do8m = rf_do8m.predict(X_test)

# Make predictions for 'do11m'
y_pred_do11m = rf_do11m.predict(X_test)

####################################################################################
# Plotting the results for 'do8m'
plt.figure(figsize=(12, 6))
plt.plot(dates_8m, y_test_do8m, color='blue', label='Observed DO 8m', marker='o')
plt.plot(dates_8m, y_pred_do8m, color='red', label='Predicted DO 8m', marker='x')
plt.title('Observed vs Predicted DO Levels at 8m Depth')
plt.xlabel('Time')
plt.ylabel('Dissolved Oxygen (mg/L)')
plt.legend()

plt.tight_layout()

# Plotting the results for 'do11m'
plt.figure(figsize=(12, 6))
plt.plot(dates_11m, y_test_do11m, color='green', label='Observed DO 11m', marker='o')
plt.plot(dates_11m, y_pred_do11m, color='orange', label='Predicted DO 11m', marker='x')
plt.title('Observed vs Predicted DO Levels at 11m Depth')
plt.xlabel('Time')
plt.ylabel('Dissolved Oxygen (mg/L)')
plt.legend()

plt.tight_layout()
plt.show()

###########################################################################################
# For 'do8m'
r2_do8m = r2_score(y_test_do8m, y_pred_do8m)
mae_do8m = mean_absolute_error(y_test_do8m, y_pred_do8m)
mse_do8m = mean_squared_error(y_test_do8m, y_pred_do8m)
rmse_do8m = mean_squared_error(y_test_do8m, y_pred_do8m, squared=False)

# For 'do11m'
r2_do11m = r2_score(y_test_do11m, y_pred_do11m)
mae_do11m = mean_absolute_error(y_test_do11m, y_pred_do11m)
mse_do11m = mean_squared_error(y_test_do11m, y_pred_do11m)
rmse_do11m = mean_squared_error(y_test_do11m, y_pred_do11m, squared=False)
print(f"Evaluation Metrics for 'do8m':")
print(f"R-squared: {r2_do8m:.2f}")
print(f"Mean Absolute Error (MAE): {mae_do8m:.2f}")
print(f"Mean Squared Error (MSE): {mse_do8m:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_do8m:.2f}\n")

print(f"Evaluation Metrics for 'do11m':")
print(f"R-squared: {r2_do11m:.2f}")
print(f"Mean Absolute Error (MAE): {mae_do11m:.2f}")
print(f"Mean Squared Error (MSE): {mse_do11m:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_do11m:.2f}")
####################################################################################
#Pierce Skill Score
#Define the Threshold and Binarize the Predicted and Actual Values
#PSS=TPR−FPR
# Define the threshold
threshold = 4

# Binarize the predicted and actual values based on the threshold
# low DO (<= 4 mg/L) is represented by 1, normal/high DO (> 4 mg/L) is represented by 0
y_test_do8m_binarized = (y_test_do8m <= threshold).astype(int)
y_pred_do8m_binarized = (y_pred_do8m <= threshold).astype(int)

y_test_do11m_binarized = (y_test_do11m <= threshold).astype(int)
y_pred_do11m_binarized = (y_pred_do11m <= threshold).astype(int)

#Calculate the Confusion Matrix, TPR, FPR, and PSS

# For 'do8m'
tn_8m, fp_8m, fn_8m, tp_8m = confusion_matrix(y_test_do8m_binarized, y_pred_do8m_binarized).ravel()

# Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for 'do8m'
tpr_8m = tp_8m / (tp_8m + fn_8m)
fpr_8m = fp_8m / (fp_8m + tn_8m)

# Calculate Peirce Skill Score (PSS) for 'do8m'
pss_8m = tpr_8m - fpr_8m

# For 'do11m'
tn_11m, fp_11m, fn_11m, tp_11m = confusion_matrix(y_test_do11m_binarized, y_pred_do11m_binarized).ravel()

# Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for 'do11m'
tpr_11m = tp_11m / (tp_11m + fn_11m)
fpr_11m = fp_11m / (fp_11m + tn_11m)

# Calculate Peirce Skill Score (PSS) for 'do11m'
pss_11m = tpr_11m - fpr_11m

print(f"Peirce Skill Score (PSS) for 'do8m': {pss_8m}")
print(f"Peirce Skill Score (PSS) for 'do11m': {pss_11m}")


# In[ ]:




