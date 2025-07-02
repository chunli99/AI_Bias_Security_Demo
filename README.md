## AI Bias in Cybersecurity: Suspicious User Detection Demo
## By Jasmione Sutton

## Project Goal
This project aims to demonstrate how algorithmic bias can inadvertently lead to unfair or discriminatory outcomes in cybersecurity systems, specifically in a simulated "suspicious user detection" model. We will illustrate how bias introduced in training data (reflecting socioeconomic disparities) can cause a model to disproportionately flag certain user groups as suspicious, even when their actual risk profiles are similar to others.

## Technologies Used
- Python
- scikit-learn (for machine learning)
- pandas (for data manipulation)
- numpy (for numerical operations)
- matplotlib/seaborn (for visualization)

## How it Works (High-Level)
1.  **Synthetic Data Generation (`generate_data.py`):** Creates a dataset of simulated user activity, including socioeconomic proxies (e.g., internet quality, device type) and derived security metrics. Bias is intentionally introduced here, making one demographic group more likely to exhibit characteristics that lead to a "fail" security outcome.
2.  **Model Training (`train_model.py`):** Loads the generated data, preprocess it, and train a machine learning classification model (e.g., Logistic Regression) to predict the `security_outcome`. The trained model is then saved.
3.  **Bias Analysis (`analyze_bias.py`):** Loads the synthetic data and the trained model. Make predictions and analyze the model's performance and fairness across different demographic groups, focusing on metrics like accuracy, precision, recall, F1-score, False Positive Rate (FPR), and False Negative Rate (FNR).
4.  **Visualization (`visualize_bias.py`):** Generates plots (e.g., bar charts of predicted fail rates, confusion matrices) to visually illustrate the disparities and the impact of bias on different demographic groups.

## Getting Started

To run this project, please follow these steps:

### 1. Clone the Repository (if applicable)
git clone <repository_url>
```Recommended name for the directory folder is: AI_Bias_Security_Demo```


### 2. Set up Python Environment
```It's recommended to use a virtual environment.```

python -m venv venv

# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate


### 3. Install Dependencies
pip install pandas numpy scikit-learn matplotlib seaborn


### 4. Configure Paths (Optional, if not using default structure)
All file paths are managed in `config.py`. The default for all output files are whever this directory lives. 
If you need to change data or model locations, modify this file.

### 5. Run the Scripts in the following order

#### a. Generate Synthetic Data
python generate_data.py

```This will create `synthetic_security_data.csv` in the project directory.```

#### b. Train the Security Model
python train_model.py

```This will train a Logistic Regression model and save it as `security_model.joblib`.```

#### c. Analyze Model Bias
python analyze_bias.py

```This will print detailed bias analysis metrics to the console.```

#### d. Visualize Bias
python visualize_bias.py

```This will generate `predicted_fail_rate_by_group.png` and `confusion_matrices_by_group.png` in the `plots/` directory.```

## Project Structure

AI_Bias_Security_Demo/
├── config.py
├── generate_data.py
├── train_model.py
├── analyze_bias.py
├── visualize_bias.py
├── README.md
├── PROGRESS.md
├── synthetic_security_data.csv (generated after running generate_data.py)
├── security_model.joblib (generated after running train_model.py)
└── plots/
    ├── predicted_fail_rate_by_group.png (generated after running visualize_bias.py)
    └── confusion_matrices_by_group.png (generated after running visualize_bias.py)
