# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:54:11 2024

@author: pablo
"""

import pandas as pd
import matplotlib.pyplot as plt
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.forecasting.base import ForecastingHorizon

def preprocess_data(filepath):
    """
    Load and preprocess the data.

    Parameters:
    filepath (str): Path to the CSV file containing the data.

    Returns:
    pd.DataFrame: Preprocessed data with date as index.
    """
    # Load the data and skip the header row
    data = pd.read_csv(filepath, header=None, names=['date', 'y'], skiprows=1)
    
    # Convert the date column to datetime format explicitly
    data['date'] = pd.to_datetime(data['date'], format='%d.%m.%y', errors='coerce')
    
    # Drop rows with NaN values
    data = data.dropna().reset_index(drop=True)
    
    # Separate date components for validation
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year

    # Check unique years, months, and days for validation purposes
    unique_years = data['year'].unique()
    unique_months_per_year = data.groupby('year')['month'].unique()
    unique_days_per_month_year = data.groupby(['year', 'month'])['day'].unique()

    print(f"Unique years: {unique_years}")
    print("Unique months per year:")
    for year, months in unique_months_per_year.items():
        print(f"  {year}: {months.tolist()}")
    print("Unique days per month and year:")
    for (year, month), days in unique_days_per_month_year.items():
        print(f"  {year}-{month}: {days.tolist()}")

    # Validate the sequence of dates
    valid_dates = True
    
    # Check if months follow the correct sequence within each year
    for year in unique_years:
        if not all(unique_months_per_year[year] == sorted(unique_months_per_year[year])):
            valid_dates = False
            print(f"Months are not in order for year {year}: {unique_months_per_year[year].tolist()}")
    
    # Check if days follow the correct sequence within each month and year
    for (year, month), days in unique_days_per_month_year.items():
        if not all(days == sorted(days)):
            valid_dates = False
            print(f"Days are not in order for year {year} and month {month}: {days.tolist()}")
    
    if valid_dates:
        print("Dates are correctly ordered and consistent.")
    else:
        print("Fix the date issues before proceeding.")
        return None

    # Set the date column as the index and ensure correct frequency
    data.set_index('date', inplace=True)
    data.index = data.index.to_period('M')
    
    return data

def process_data(data):
    """
    Process the data to train the model and make predictions.

    Parameters:
    data (pd.DataFrame): Preprocessed data with date as index.

    Returns:
    tuple: Training data, test data, predicted test data, forecasted data.
    """
    # Extract the target variable
    y = data['y']
    
    # Split data into train and test sets (last 12 points for validation)
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    
    # Define the forecasting horizon for the test set
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    
    # Initialize and fit the model on the training data
    model = AutoARIMA(sp=12, suppress_warnings=True)
    model.fit(y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(fh)
    
    # Evaluate the model's performance
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f'MAPE: {mape}')
    
    # Refit the model on the entire dataset for forecasting the next 12 months
    model.fit(y)
    
    # Forecast the next 12 months
    fh_future = pd.period_range(start='2021-03', periods=12, freq='M')
    y_forecast = model.predict(fh=fh_future)
    
    # Convert PeriodIndex to Timestamp for saving
    forecast_df = pd.DataFrame(y_forecast, columns=['y'])
    forecast_df.index = fh_future.to_timestamp()
    forecast_df.to_csv('test.csv', index_label='date')
    
    print("Forecast saved to test.csv")
    
    return y_train, y_test, y_pred, forecast_df

def postprocess_data(y_train, y_test, y_pred, forecast_df):
    """
    Post-process the data to visualize the results.

    Parameters:
    y_train (pd.Series): Training data.
    y_test (pd.Series): Test data.
    y_pred (pd.Series): Predicted test data.
    forecast_df (pd.DataFrame): Forecasted data.
    """
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index.to_timestamp(), y_train, label='Training Data')
    plt.plot(y_test.index.to_timestamp(), y_test, label='Test Data', color='orange')
    plt.plot(y_test.index.to_timestamp(), y_pred, label='Predicted Test Data', color='green')
    plt.plot(forecast_df.index, forecast_df['y'], label='Forecasted Data', color='red')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Time Series Forecasting')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to run the entire process.
    """
    filepath = 'train.csv'
    data = preprocess_data(filepath)
    if data is not None:
        y_train, y_test, y_pred, forecast_df = process_data(data)
        postprocess_data(y_train, y_test, y_pred, forecast_df)

if __name__ == "__main__":
    main()
