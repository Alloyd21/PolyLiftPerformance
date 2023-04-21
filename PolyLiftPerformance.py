import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    print("Loading CSV file...")
    df = pd.read_csv('/Users/adamlloyd/Downloads/openpowerlifting.csv', low_memory=False)
    print("CSV file loaded.")
    df['BodyweightKg'] = pd.to_numeric(df['BodyweightKg'])
    df['TotalKg'] = pd.to_numeric(df['TotalKg'])
    df = df.dropna(subset=['BodyweightKg', 'TotalKg'])
    print(f"Number of rows after dropping missing values: {len(df):,}")

    X = df['BodyweightKg'].values.reshape(-1, 1)
    y = df['TotalKg'].values
    poly, model = train_polynomial_regression(X, y)
    return poly, model

def train_polynomial_regression(X, y, degree=3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    return poly, model

def calculate_custom_points(body_weight, total_lifted, model, poly):
    body_weight_poly = poly.transform([[body_weight]])
    predicted_total = model.predict(body_weight_poly)[0]
    custom_points = total_lifted / predicted_total * 500
    return custom_points

if __name__ == '__main__':
    poly, model = main()
    body_weight = float(input("Enter body weight in kilograms: "))
    total_lifted = float(input("Enter total weight lifted (squat + bench press + deadlift) in kilograms: "))
    custom_points = calculate_custom_points(body_weight, total_lifted, model, poly)
    print(f"Custom Points: {custom_points:.2f}")
