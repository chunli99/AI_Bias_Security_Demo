import pandas as pd
import numpy as np
import random

def generate_synthetic_security_data(num_users=1000, group_b_proportion=0.3, bias_strength=0.7):
    """
    Author: Jasmine Sutton
    Date: July 2025
    Purpose: This is to generate sample data with an bias against a group of individuals who have 
             slower and inconsistent connectivity methods, that may be incorrectly flagged for security issues.
             This code generates synthetic security data with intentional bias against 'Group B'.

    Args:
        num_users (int): Total number of users to generate.
        group_b_proportion (float): Proportion of users belonging to 'Group B'.
        bias_strength (float): How strongly 'Group B' is pushed towards biased features (0.0 to 1.0).
                                1.0 means Group B exclusively gets biased features.
    Returns:
        pd.DataFrame: A DataFrame containing the synthetic security data.
    """

    data = []
    for i in range(num_users):
        user_id = f'user_{i+1}'
        
        # Assign demographic group
        demographic_group = 'Group B' if random.random() < group_b_proportion else 'Group A'

        # --- Socioeconomic Proxies (biased distribution) ---
        internet_access_quality = ''
        device_type = ''

        if demographic_group == 'Group A':
            # Group A: More likely to have high-quality internet and modern devices
            internet_access_quality = random.choices(
                ['High-Speed Home', 'Standard Home', 'Mobile Hotspot', 'Public Wi-Fi'],
                weights=[0.6, 0.3, 0.07, 0.03], k=1
            )[0]
            device_type = random.choices(
                ['Modern Personal', 'Older Personal', 'Shared/Public'],
                weights=[0.7, 0.25, 0.05], k=1
            )[0]
        else: # Group B
            # Group B: More likely to have lower-quality internet and older/shared devices
            internet_access_quality = random.choices(
                ['High-Speed Home', 'Standard Home', 'Mobile Hotspot', 'Public Wi-Fi'],
                weights=[0.1 * (1-bias_strength), 0.2 * (1-bias_strength), 0.3 + 0.3*bias_strength, 0.4 + 0.4*bias_strength], k=1
            )[0]
            device_type = random.choices(
                ['Modern Personal', 'Older Personal', 'Shared/Public'],
                weights=[0.1 * (1-bias_strength), 0.4 + 0.4*bias_strength, 0.5 + 0.5*bias_strength], k=1
            )[0]

        # --- Derived Security Metrics (influenced by socioeconomic proxies) ---
        network_latency = 0
        if internet_access_quality == 'High-Speed Home':
            network_latency = np.random.normal(30, 10) # Mean 30ms
        elif internet_access_quality == 'Standard Home':
            network_latency = np.random.normal(70, 20) # Mean 70ms
        elif internet_access_quality == 'Mobile Hotspot':
            network_latency = np.random.normal(150, 50) # Mean 150ms
        elif internet_access_quality == 'Public Wi-Fi':
            network_latency = np.random.normal(250, 80) # Mean 250ms
        network_latency = max(5, int(network_latency)) # Ensure positive

        device_age = 0
        if device_type == 'Modern Personal':
            device_age = np.random.uniform(0, 24) # 0-24 months
        elif device_type == 'Older Personal':
            device_age = np.random.uniform(24, 72) # 24-72 months
        elif device_type == 'Shared/Public':
            device_age = np.random.uniform(60, 120) # 60-120 months
        device_age = max(1, int(device_age)) # Ensure positive

        software_update_frequency = 0
        if device_type == 'Modern Personal':
            software_update_frequency = np.random.normal(7, 3) # Every 7 days
        elif device_type == 'Older Personal':
            software_update_frequency = np.random.normal(20, 7) # Every 20 days
        elif device_type == 'Shared/Public':
            software_update_frequency = np.random.normal(45, 15) # Every 45 days
        software_update_frequency = max(1, int(software_update_frequency)) # Ensure positive

        # Access requests per day (less directly tied to socioeconomic, but can have slight group bias)
        access_requests_per_day = np.random.poisson(5) # Baseline
        if demographic_group == 'Group B':
            access_requests_per_day = np.random.poisson(7) # Slightly higher for Group B
        access_requests_per_day = max(1, int(access_requests_per_day)) # Ensure positive

        # --- Determine Security Outcome (based on thresholds) ---
        # Higher values for latency, age, frequency, requests are 'worse'
        security_outcome = 0 # Default to Pass

        # Thresholds (can be tuned)
        if network_latency > 100:
            security_outcome = 1
        if device_age > 48:
            security_outcome = 1
        if software_update_frequency > 30:
            security_outcome = 1
        if access_requests_per_day > 10:
            security_outcome = 1
        
        # Introduce a small amount of random noise to outcomes to make it less deterministic
        if random.random() < 0.05: # 5% chance to flip outcome
            security_outcome = 1 - security_outcome

        data.append([
            user_id, demographic_group, internet_access_quality, device_type,
            network_latency, device_age, software_update_frequency,
            access_requests_per_day, security_outcome
        ])

    df = pd.DataFrame(data, columns=[
        'user_id', 'demographic_group', 'internet_access_quality', 'device_type',
        'network_latency', 'device_age', 'software_update_frequency',
        'access_requests_per_day', 'security_outcome'
    ])
    return df

if __name__ == "__main__":
    import sys
    import os
    # Add the current script's directory to the Python path to find config.py
    script_dir = os.path.dirname(__file__)
    sys.path.append(script_dir)
    from config import SYNTHETIC_DATA_PATH
    df = generate_synthetic_security_data(num_users=2000, group_b_proportion=0.3, bias_strength=0.8)
    df.to_csv(SYNTHETIC_DATA_PATH, index=False)
    print(f"Synthetic data generated and saved to {SYNTHETIC_DATA_PATH}")
    print(df.head())
    print("\nValue counts for demographic_group:")
    print(df['demographic_group'].value_counts())
    print("\nValue counts for security_outcome:")
    print(df['security_outcome'].value_counts())
    print("\nMean security_outcome by demographic_group:")
    print(df.groupby('demographic_group')['security_outcome'].mean())