# AI Bias in Cybersecurity: Project Progress

## Planned Steps:
1.  **Setup Project Environment:** Create project directory, README, and PROGRESS files. (DONE)
2.  **Data Generation:** Write Python script to generate synthetic user data with demographic proxies and simulate a biased security metric/outcome. (DONE)
3.  **Model Training:** Implement a simple ML model to predict security outcomes. (DONE)
4.  **Bias Analysis:** Evaluate model fairness across different demographic groups. (DONE)
5.  **Visualization:** Create plots to illustrate the bias. (DONE)
6.  **Refinement & Documentation:** Clean code, update README, add comments, centralize paths in `config.py`. (DONE)

## Accomplished So Far:
- Created `AI_Bias_Security_Demo` directory.
- Created `README.md` with project goal and high-level overview.
- Created `PROGRESS.md` to track steps.
- Refined data generation scope to include demographic proxies and biased security metrics.
- Wrote `generate_data.py` script to create synthetic security data with intentional bias.
- Wrote `train_model.py` script to load data, preprocess, train a Logistic Regression model, and save it.
- Wrote `analyze_bias.py` script to load data and model, make predictions, and perform bias analysis.
- Wrote `visualize_bias.py` script to generate plots illustrating the bias.
- Created `config.py` for centralized path management.
- Updated all scripts (`generate_data.py`, `train_model.py`, `analyze_bias.py`, `visualize_bias.py`) to use paths from `config.py`.
- **FIXED:** Added `sys.path.append(os.path.dirname(__file__))` to each script's `if __name__ == "__main__":` block to ensure `config.py` is found.
- Updated `README.md` with detailed instructions and project structure.