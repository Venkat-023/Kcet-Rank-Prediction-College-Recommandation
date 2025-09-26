# streamlit_app.py

import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Load trained model and scaler
# -----------------------------
model_path = "best_tree_model.pkl"  # Path to your trained Gradient Boosting model
scaler_path = "scaler.pkl"          # Path to your trained StandardScaler

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# Hardcoded total students per year
# -----------------------------
total_students_per_year = {
    2020: 120000,
    2021: 125000,
    2022: 130000,
    2023: 244000,
    2024: 310000,
    2025: 312000,
    2026: 318000
}

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="KCET Rank Predictor",
    page_icon="🛸",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Sci-Fi UI Header
# -----------------------------
st.markdown(
    """
    <div style="background-color:#0f111a;padding:15px;border-radius:10px">
    <h1 style="color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;">
        🚀 KCET Rank Predictor 3000
    </h1>
    <p style="color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;">
        Enter your scores below to see your predicted rank
    </p>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# User Input
# -----------------------------
with st.form("score_form"):
    st.markdown("### Input Your Marks")
    
    kcet_marks = st.number_input("KCET Marks (out of 180)", min_value=0, max_value=180, step=1)
    board_marks = st.number_input("Board Marks (out of 300)", min_value=0, max_value=300, step=1)
    year = st.selectbox("Exam Year", options=list(total_students_per_year.keys()))
    
    submitted = st.form_submit_button("Predict Rank 🚀")

if submitted:
    # -----------------------------
    # Calculate percentages
    # -----------------------------
    kcet_percent = (kcet_marks / 180) * 100
    board_percent = (board_marks / 300) * 100
    
    # Weighted score (50% KCET + 50% Board)
    score = (kcet_percent + board_percent) / 2.0
    
    # Total students
    total_students = total_students_per_year[year]
    
    # -----------------------------
    # Prepare features for model
    # -----------------------------
    # Model expects: [score, year, total_students]
    X_input = np.array([[score, year, total_students]], dtype=float)
    
    # Apply StandardScaler
    X_input_scaled = scaler.transform(X_input)
    
    # Predict rank
    predicted_rank = model.predict(X_input_scaled)[0]
    
    # -----------------------------
    # Display Calculated Scores
    # -----------------------------
    st.markdown(
        f"""
        <div style="background-color:#0f111a;padding:20px;border-radius:10px;margin-top:15px">
        <h2 style="color:#ff3c00;font-family:Orbitron, sans-serif;text-align:center;">
            🎯 Calculated Scores
        </h2>
        <p style="color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;">
            KCET %: {kcet_percent:.2f}%<br>
            Board %: {board_percent:.2f}%<br>
            Weighted Score: {score:.2f}%
        </p>
        </div>
        """, unsafe_allow_html=True
    )
    
    # -----------------------------
    # Display Predicted Rank
    # -----------------------------
    st.markdown(
        f"""
        <div style="background-color:#0f111a;padding:20px;border-radius:10px;margin-top:15px">
        <h2 style="color:#ff3c00;font-family:Orbitron, sans-serif;text-align:center;">
            🛸 Predicted Rank: {int(predicted_rank)}
        </h2>
        <p style="color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;">
            Based on year: {year} and total students: {total_students}
        </p>
        </div>
        """, unsafe_allow_html=True
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;'>Made with 💫 by Your Team</p>",
    unsafe_allow_html=True
)

