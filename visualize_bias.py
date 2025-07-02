import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def visualize_model_bias(data_path, model_path, output_dir="./plots"):
    """
    Author: Jasmine Sutton
    Date: July 2025
    Purpose: This code loads the synthetic data and trained model, makes predictions, and visualizes bias
             across different demographic groups.

    
    Args:
        data_path (str): Path to the synthetic security data CSV file.
        model_path (str): Path to the trained model joblib file.
        output_dir (str): Directory to save the generated plots.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)

    # Load the trained model pipeline
    model_pipeline = joblib.load(model_path)

    # Define features (X) and target (y)
    X = df.drop(['user_id', 'security_outcome'], axis=1)
    y_true = df['security_outcome']

    # Make predictions using the loaded model
    y_pred = model_pipeline.predict(X)

    # Add predictions to the DataFrame for easier analysis
    df['predicted_security_outcome'] = y_pred

    # --- Visualization 1: Predicted Fail Rate by Demographic Group ---
    plt.figure(figsize=(8, 6))
    sns.barplot(x='demographic_group', y='predicted_security_outcome', data=df, estimator=lambda x: sum(x)/len(x) * 100)
    plt.title('Predicted Security Fail Rate by Demographic Group')
    plt.ylabel('Predicted Fail Rate (%)')
    plt.xlabel('Demographic Group')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'predicted_fail_rate_by_group.png'))
    plt.close()
    print(f"Saved predicted_fail_rate_by_group.png to {output_dir}")

    # --- Visualization 2: Confusion Matrices by Demographic Group ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Confusion Matrices by Demographic Group')

    for i, group in enumerate(df['demographic_group'].unique()):
        group_df = df[df['demographic_group'] == group]
        group_y_true = group_df['security_outcome']
        group_y_pred = group_df['predicted_security_outcome']

        cm = confusion_matrix(group_y_true, group_y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                    xticklabels=['Predicted Pass', 'Predicted Fail'],
                    yticklabels=['Actual Pass', 'Actual Fail'])
        axes[i].set_title(f'Group: {group}')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_by_group.png'))
    plt.close()
    print(f"Saved confusion_matrices_by_group.png to {output_dir}")

if __name__ == "__main__":
    import sys
    import os
    # Add the current script's directory to the Python path to find config.py
    script_dir = os.path.dirname(__file__)
    sys.path.append(script_dir)
    from config import SYNTHETIC_DATA_PATH, MODEL_PATH, PLOTS_DIR
    visualize_model_bias(SYNTHETIC_DATA_PATH, MODEL_PATH, PLOTS_DIR)
