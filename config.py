# Centralized configuration for the AI Bias in Cybersecurity Demo project
#By Jasmine Sutton
#Date: July 1, 2025

import os

# Base directory for the project (current directory of this config file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
SYNTHETIC_DATA_PATH = os.path.join(BASE_DIR, "synthetic_security_data.csv")

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "security_model.joblib")

# Plot output directory
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Ensure the plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)
