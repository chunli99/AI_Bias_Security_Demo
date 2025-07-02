import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def analyze_model_bias(data_path, model_path):
    """
    Author: Jasmine Sutton
    Date: July, 2025
    Purpose: Loads the synthetic data and trained model, makes predictions, and analyzes bias
             across different demographic groups.

    Args:
        data_path (str): Path to the synthetic security data CSV file.
        model_path (str): Path to the trained model joblib file.
    """
    # Load data
    df = pd.read_csv(data_path)

    # Load the trained model pipeline
    model_pipeline = joblib.load(model_path)

    # Define features (X) and target (y)
    X = df.drop(['user_id', 'security_outcome'], axis=1)
    y_true = df['security_outcome']

    # Make predictions using the loaded model
    print("Making predictions...")
    y_pred = model_pipeline.predict(X)
    print("Predictions complete.")

    # Add predictions to the DataFrame for easier analysis
    df['predicted_security_outcome'] = y_pred

    print("\n--- Overall Model Performance ---")
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred)
    overall_recall = recall_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1-Score: {overall_f1:.4f}")

    print("\n--- Bias Analysis by Demographic Group ---")
    for group in df['demographic_group'].unique():
        group_df = df[df['demographic_group'] == group]
        group_y_true = group_df['security_outcome']
        group_y_pred = group_df['predicted_security_outcome']

        group_accuracy = accuracy_score(group_y_true, group_y_pred)
        group_precision = precision_score(group_y_true, group_y_pred, zero_division=0)
        group_recall = recall_score(group_y_true, group_y_pred, zero_division=0)
        group_f1 = f1_score(group_y_true, group_y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
        
        # False Positive Rate (FPR): FP / (FP + TN) - Proportion of actual negatives incorrectly classified as positive
        # False Negative Rate (FNR): FN / (FN + TP) - Proportion of actual positives incorrectly classified as negative
        group_fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        group_fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

        print(f"\n--- Group: {group} ---")
        print(f"  Count: {len(group_df)}")
        print(f"  Actual Fail Rate: {group_y_true.mean():.4f}")
        print(f"  Predicted Fail Rate: {group_y_pred.mean():.4f}")
        print(f"  Accuracy: {group_accuracy:.4f}")
        print(f"  Precision: {group_precision:.4f}")
        print(f"  Recall: {group_recall:.4f}")
        print(f"  F1-Score: {group_f1:.4f}")
        print(f"  False Positive Rate (FPR): {group_fpr:.4f}")
        print(f"  False Negative Rate (FNR): {group_fnr:.4f}")

if __name__ == "__main__":
    import sys
    import os
    # Add the current script's directory to the Python path to find config.py
    script_dir = os.path.dirname(__file__)
    sys.path.append(script_dir)
    from config import SYNTHETIC_DATA_PATH, MODEL_PATH
    analyze_model_bias(SYNTHETIC_DATA_PATH, MODEL_PATH)
