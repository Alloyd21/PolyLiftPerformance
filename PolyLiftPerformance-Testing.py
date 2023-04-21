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

    print("Training gender-agnostic model...")
    poly, model = train_polynomial_regression(df)
    print("\nTraining male model...")
    df_male = df[df['Sex'] == 'M']
    poly_male, model_male = train_polynomial_regression(df_male)
    print("\nTraining female model...")
    df_female = df[df['Sex'] == 'F']
    poly_female, model_female = train_polynomial_regression(df_female)

    return poly, model, poly_male, model_male, poly_female, model_female

def train_polynomial_regression(df, degree=3):
    X = df['BodyweightKg'].values.reshape(-1, 1)
    y = df['TotalKg'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"Train Mean Squared Error: {mse_train}")
    print(f"Test Mean Squared Error: {mse_test}")
    print(f"Train R2 Score: {r2_train}")
    print(f"Test R2 Score: {r2_test}")
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {model.coef_}")

    return poly, model

def calculate_custom_points(gender, body_weight, total_lifted, model_male, poly_male, model_female, poly_female, model_agnostic, poly_agnostic):
    if gender.lower() == 'male':
        model = model_male
        poly = poly_male
    elif gender.lower() == 'female':
        model = model_female
        poly = poly_female
    elif gender.lower() == 'agnostic':
        model = model_agnostic
        poly = poly_agnostic
    else:
        raise ValueError("Invalid gender. Please enter 'male', 'female', or 'agnostic'.")

    body_weight_poly = poly.transform([[body_weight]])
    predicted_total = model.predict(body_weight_poly)[0]
    custom_points = total_lifted / predicted_total * 500
    return custom_points

if __name__ == '__main__':
    poly, model, poly_male, model_male, poly_female, model_female = main()

    gender = input("Enter gender (male, female, or agnostic): ")
    body_weight = float(input("Enter body weight in kilograms: "))
    total_lifted = float(input("Enter total weight lifted (squat + bench press + deadlift) in kilograms: "))

    custom_points = calculate_custom_points(gender, body_weight, total_lifted, model_male, poly_male, model_female, poly_female, model, poly)
    print(f"Custom Points: {custom_points:.2f}")

    if gender.lower() in ['male', 'female']:
        agnostic_points = calculate_custom_points('agnostic', body_weight, total_lifted, model_male, poly_male, model_female, poly_female, model, poly)
        print(f"Gender-Agnostic Points: {agnostic_points:.2f}")
