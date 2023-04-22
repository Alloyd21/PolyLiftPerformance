import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    print("\n Loading CSV file...")
    df = pd.read_csv('openpowerlifting.csv', low_memory=False)
    print(f"{len(df):,} rows loaded")
    df['BodyweightKg'] = pd.to_numeric(df['BodyweightKg'])
    df['TotalKg'] = pd.to_numeric(df['TotalKg'])
    df = df.dropna(subset=['BodyweightKg', 'TotalKg'])
    print(f"{len(df):,} rows after dropping missing values ")

    # Calculate the mean and standard deviation for the columns
    bodyweight_mean = df['BodyweightKg'].mean()
    bodyweight_std = df['BodyweightKg'].std()
    total_mean = df['TotalKg'].mean()
    total_std = df['TotalKg'].std()

    # Create a boolean mask to filter out rows outside 3 standard deviations
    mask = (
        (df['BodyweightKg'] > bodyweight_mean - 3 * bodyweight_std) &
        (df['BodyweightKg'] < bodyweight_mean + 3 * bodyweight_std) &
        (df['TotalKg'] > total_mean - 3 * total_std) &
        (df['TotalKg'] < total_mean + 3 * total_std)
    )
    df = df[mask]
    print(f"{len(df):,} rows after removing outliers ")

    # Separate male and female data
    df_male = df[df['Sex'] == 'M']
    df_female = df[df['Sex'] == 'F']

    # Undersample the male data
    df_male_undersampled = df_male.sample(n=len(df_female), random_state=42)

    # Combine the undersampled male and female data
    df_balanced = pd.concat([df_male_undersampled, df_female])
    total_len = len(df_male_undersampled) + len(df_female)
    print(f"{total_len:,} rows after male undersampling: ")

    print("\n Training gender-agnostic model...")
    poly_balanced, model_balanced = train_polynomial_regression(df_balanced)
    print("\nTraining male model...")
    poly_male_balanced, model_male_balanced = train_polynomial_regression(
        df_male_undersampled)
    print("\nTraining female model...")
    poly_female_balanced, model_female_balanced = train_polynomial_regression(
        df_female)

    return poly_balanced, model_balanced, poly_male_balanced, model_male_balanced, poly_female_balanced, model_female_balanced, df_balanced, df_male_undersampled, df_female

def train_polynomial_regression(df, degree=3):
    X = df['BodyweightKg'].values.reshape(-1, 1)
    y = df['TotalKg'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
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
        raise ValueError(
            "Invalid gender. Please enter 'male', 'female', or 'agnostic'.")

    body_weight_poly = poly.transform([[body_weight]])
    predicted_total = model.predict(body_weight_poly)[0]
    custom_points = total_lifted / predicted_total * 500
    return custom_points

def plot_data_and_models(df, df_male_undersampled, df_female, poly, model, poly_male, model_male, poly_female, model_female):
    plt.figure(figsize=(25, 15))
    X = np.linspace(df['BodyweightKg'].min(),
                    df['BodyweightKg'].max(), num=300)
    X_poly = poly.transform(X.reshape(-1, 1))
    X_poly_male = poly_male.transform(X.reshape(-1, 1))
    X_poly_female = poly_female.transform(X.reshape(-1, 1))

    y_pred = model.predict(X_poly)
    y_pred_male = model_male.predict(X_poly_male)
    y_pred_female = model_female.predict(X_poly_female)

    plt.scatter(df_male_undersampled['BodyweightKg'],
                df_male_undersampled['TotalKg'], alpha=0.2, s=5, color='red')
    plt.scatter(df_female['BodyweightKg'],
                df_female['TotalKg'], alpha=0.2, s=5, color='green')
    plt.plot(X, y_pred, label='Agnostic', color='blue')
    plt.plot(X, y_pred_male, label='Male', color='red')
    plt.plot(X, y_pred_female, label='Female', color='green')
    plt.xlabel('Body Weight (kg)')
    plt.ylabel('Total Lifted (kg)')
    plt.title('Body Weight vs. Total Lifted Weight')
    plt.legend()
    plt.savefig('bodyweight_vs_total_lifted_1500x1500.png', dpi=150)

if __name__ == '__main__':
    poly, model, poly_male, model_male, poly_female, model_female, df, df_male_undersampled, df_female = main()
    plot_data_and_models(df, df_male_undersampled, df_female, poly,
                         model, poly_male, model_male, poly_female, model_female)

    gender = input("\n Enter gender (male, female, or agnostic): ")
    body_weight = float(input("Enter body weight in kilograms: "))
    total_lifted = float(input(
        "Enter total weight lifted (squat + bench press + deadlift) in kilograms: "))

    custom_points = calculate_custom_points(
        gender, body_weight, total_lifted, model_male, poly_male, model_female, poly_female, model, poly)
    print(f" \n Custom Points {gender}: {custom_points:.2f}")

    if gender.lower() in ['male', 'female']:
        agnostic_points = calculate_custom_points(
            'agnostic', body_weight, total_lifted, model_male, poly_male, model_female, poly_female, model, poly)
        print(f"Gender-Agnostic Points: {agnostic_points:.2f} \n")
