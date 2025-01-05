import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath).iloc[:, 1:]
    headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors",
               "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width",
               "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size",
               "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
               "city-mpg", "highway-mpg", "price"]
    df.columns = headers
    df.replace('?', np.nan, inplace=True)
    numeric_columns = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['price'], inplace=True)
    df['city-L/100km'] = 235 / df['city-mpg']
    df.drop(columns=['city-mpg'], inplace=True)
    df['price'] = df['price'].astype(float)
    return df


def preprocess_data(df):
    # Handle missing values
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_categorical = SimpleImputer(strategy='most_frequent')

    # Separate numeric and categorical features
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Apply imputers
    df[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])
    df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    return df


def train_and_evaluate_models(df):
    # Separate features and target
    X = df.drop(columns=['price'])
    y = df['price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        "Linear_Regression": LinearRegression(),
        "Random_Forest": RandomForestRegressor(random_state=42),
        "Decision_Tree": DecisionTreeRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
    }

    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Save model to file
        filename = f"{name}_model.pkl"
        joblib.dump(model, filename)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "R^2": r2}

    return results


if __name__ == '__main__':
    filepath = 'output.csv'
    df = load_and_preprocess_data(filepath)
    df = preprocess_data(df)
    results = train_and_evaluate_models(df)
    print("Model evaluation results:")
    for model, metrics in results.items():
        print(f"{model}: MSE = {metrics['MSE']:.2f}, R^2 = {metrics['R^2']:.2f}")
