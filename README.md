KCET Rank Prediction and College Recommendation System
Project Overview
This project presents a robust machine learning solution to predict KCET (Karnataka Common Entrance Test) ranks based on marks and provide personalized college recommendations. The system aids students in estimating their competitive rank prior to official results and assists in selecting suitable colleges based on predicted ranks and branch preferences. It leverages comprehensive official datasets, advanced regression models, and an intuitive web interface deployed via Streamlit.

Dataset Acquisition and Preparation
Sources: Official KCET datasets were gathered from the KEA website for the years 2023, 2024, and 2025, capturing marks, ranks, and related student performance data.

Data Cleaning: The most challenging aspect involved converting raw data from diverse formats into unified Excel files, handling missing values, ensuring consistency across years, and normalizing columns.

Integration: The cleaned datasets were merged to build a comprehensive data foundation for accurate modeling.

Exploratory Data Analysis
Performed in-depth data visualization and analysis including:

Distribution plots to understand marks and rank spread.

Correlation heatmaps to identify relationships between features.

Statistical summaries highlighting data quality and potential biases.

Machine Learning Methodology
Models and Hyperparameter Tuning
Implemented and compared several state-of-the-art regression models using rigorous hyperparameter tuning via Randomized Search CV with cross-validation:

Model	Best Performance (R²)	RMSE	Key Hyperparameters Example
Gradient Boosting	0.9996	1169.60	subsample=0.8, n_estimators=500
Random Forest	0.9996	1170.07	n_estimators=100, max_depth=30
Decision Tree	0.9996	1170.01	min_samples_split=8, max_depth=40
Extra Trees	0.9994	1345.15	n_estimators=100, max_depth=50
Ridge Regression	0.9214	15872.43	alpha=0.01, solver='lsqr'
Gradient Boosting emerged as the best performing model based on R² and RMSE.

Regularized linear models provided strong baselines, confirming the non-linear models’ superior fit.

Evaluation Metrics
R² Score: Coefficient of determination measuring explained variance.

Root Mean Squared Error (RMSE): Average prediction error magnitude in rank units.

Streamlit Web Application
Features
Interactive input for KCET marks, board marks, and exam year selection.

Real-time rank prediction displayed alongside calculated mark percentages.

Sci-fi themed UI with Orbitron font and custom CSS styling.

Responsive layout suitable for desktop and mobile.

Deployment
Deployed on Render for easy web accessibility.

Codebase includes the trained Gradient Boosting model and StandardScaler, bundled for prediction consistency.

Users can input their marks to receive instant rank estimates and explore college recommendations.

Running Locally
Clone the repository:

git clone https://github.com/yourusername/kcet-rank-prediction.git
cd kcet-rank-prediction
Install dependencies:

pip install -r requirements.txt
Launch the app:

streamlit run streamlit_app.py
Future Enhancements
Incorporate category-wise seat reservations and quota-aware recommendations.

Extend dataset with additional years and branch-specific cut-off trends.

Deploy advanced interpretability techniques such as SHAP to explain model predictions.

Integrate multi-modal input options including image and voice input for enhanced accessibility.

Repository Structure

/kcet-rank-prediction
│
├── streamlit_app.py           # Streamlit web app code
├── best_tree_model.pkl        # Trained Gradient Boosting model
├── scaler.pkl                 # Feature scaler
├── data/                      # Raw and cleaned datasets
├── notebooks/                 # Data analysis and model training explorations
├── requirements.txt           # Python package dependencies
└── README.md                  # This project description
