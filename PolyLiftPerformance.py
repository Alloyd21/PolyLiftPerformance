import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    # Load the dataset from a CSV file
    print("Loading CSV file...")
    df = pd.read_csv('/Users/adamlloyd/Downloads/openpowerlifting.csv', low_memory=False)
    print("CSV file loaded.")

    # Convert the BodyweightKg and TotalKg columns to numeric
    df['BodyweightKg'] = pd.to_numeric(df['BodyweightKg'])
    df['TotalKg'] = pd.to_numeric(df['TotalKg'])

    # Drop rows with missing values in 'BodyweightKg' or 'TotalKg' columns
    df = df.dropna(subset=['BodyweightKg', 'TotalKg'])
    print(f"Number of rows after dropping missing values: {len(df):,}")

    # Extract the relevant data
    X = df['BodyweightKg'].values.reshape(-1, 1)
    y = df['TotalKg'].values

    # Train the polynomial regression model
    poly, model = train_polynomial_regression(X, y)
    return poly, model

def train_polynomial_regression(X, y, degree=3):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the PolynomialFeatures instance and transform the data
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Fit the LinearRegression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    return poly, model

def calculate_custom_points(body_weight, total_lifted, model, poly):
    # Transform the bodyweight value and make a prediction
    body_weight_poly = poly.transform([[body_weight]])
    predicted_total = model.predict(body_weight_poly)[0]

    # Calculate the points based on the ratio between the actual and predicted total
    custom_points = total_lifted / predicted_total * 500

    return custom_points

if __name__ == '__main__':
    poly, model = main()
    body_weight = float(input("Enter body weight in kilograms: "))
    total_lifted = float(input("Enter total weight lifted (squat + bench press + deadlift) in kilograms: "))
    custom_points = calculate_custom_points(body_weight, total_lifted, model, poly)
    print(f"Custom Points: {custom_points:.2f}")
