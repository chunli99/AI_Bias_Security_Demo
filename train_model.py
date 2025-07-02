import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def train_security_model(data_path, model_output_path):
    """
    Author: Jasmine Sutton
    Date: July 2025
    Purpose: Loads synthetic security data, preprocesses it, trains a Logistic Regression model,
             and saves the trained model.

    Args:
        data_path (str): Path to the synthetic security data CSV file.
        model_output_path (str): Path to save the trained model.
    """
    # Load data
    df = pd.read_csv(data_path)

    # Define features (X) and target (y)
    # We exclude 'user_id' as it's just an identifier
    # 'demographic_group' is a feature we want to analyze bias against, so it's an input feature
    X = df.drop(['user_id', 'security_outcome'], axis=1)
    y = df['security_outcome']

    # Identify categorical and numerical features
    categorical_features = ['demographic_group', 'internet_access_quality', 'device_type']
    numerical_features = ['network_latency', 'device_age', 'software_update_frequency', 'access_requests_per_day']

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create a pipeline that first preprocesses and then trains the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42))
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    print("Training model...")
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate model (optional, but good for sanity check)
    train_accuracy = model_pipeline.score(X_train, y_train)
    test_accuracy = model_pipeline.score(X_test, y_test)
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save the trained model
    joblib.dump(model_pipeline, model_output_path)
    print(f"Trained model saved to {model_output_path}")

if __name__ == "__main__":
    import sys
    import os
    # Add the current script's directory to the Python path to find config.py
    script_dir = os.path.dirname(__file__)
    sys.path.append(script_dir)
    from config import SYNTHETIC_DATA_PATH, MODEL_PATH
    train_security_model(SYNTHETIC_DATA_PATH, MODEL_PATH)
