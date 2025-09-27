import streamlit as st
import numpy as np
import pickle
import pandas as pd

# -----------------------------
# Load trained model and scaler
# -----------------------------
model_path = "best_tree_model.pkl"
scaler_path = "scaler.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# Load college database Excel
# -----------------------------
college_df = pd.read_excel("colleges_list.xlsx")

# Ensure GM column is numeric (invalid entries -> 0)
college_df["GM"] = pd.to_numeric(college_df["GM"], errors="coerce").fillna(0).astype(int)

# -----------------------------
# Total students per year (hardcoded)
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
# Streamlit Page Config and Header
# -----------------------------
st.set_page_config(
    page_title="KCET Rank Predictor",
    page_icon="🛸",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <div style="background-color:#0f111a;padding:15px;border-radius:10px">
    <h1 style="color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;">
        🚀 KCET Rank Predictor & College Recommender 3000
    </h1>
    <p style="color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;">
        Enter your scores below to see your predicted rank and potential colleges
    </p>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# User Input Form
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
    score = (kcet_percent + board_percent) / 2.0
    
    total_students = total_students_per_year[year]
    
    # -----------------------------
    # Prepare features and predict
    # -----------------------------
    X_input = np.array([[score, year, total_students]], dtype=float)
    X_input_scaled = scaler.transform(X_input)
    predicted_rank = model.predict(X_input_scaled)[0]
    int_predicted_rank = int(predicted_rank)
    
    # -----------------------------
    # Display Calculated Scores & Predicted Rank
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
    
    st.markdown(
        f"""
        <div style="background-color:#0f111a;padding:20px;border-radius:10px;margin-top:15px">
        <h2 style="color:#ff3c00;font-family:Orbitron, sans-serif;text-align:center;">
            🛸 Predicted Rank: {int_predicted_rank}
        </h2>
        <p style="color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;">
            Based on exam year: {year} and total students: {total_students}
        </p>
        </div>
        """, unsafe_allow_html=True
    )

    # -----------------------------
    # Filter Eligible Colleges based on predicted rank
    # -----------------------------
    eligible_colleges = (
        college_df[college_df['GM'] >= int_predicted_rank]  # filter by GM
        .sort_values('GM')                                   # sort by GM
        .reset_index(drop=True)
    )
    
    # -----------------------------
    # Display college recommendations
    # -----------------------------
    st.markdown(
        f"""
        <div style="background-color:#0f111a;padding:20px;border-radius:10px;margin-top:20px">
        <h2 style="color:#ff3c00;font-family:Orbitron, sans-serif;text-align:center;">
            🎓 Colleges You Can Get Into (GM ≥ {int_predicted_rank})
        </h2>
        </div>
        """, unsafe_allow_html=True
    )
    
    if eligible_colleges.empty:
        st.warning("No colleges found that match your predicted rank. Try adjusting your inputs or checking back later.")
    else:
        # Display the table with custom styling
        styled_df = eligible_colleges.style.set_table_styles(
            [
                {'selector': 'th', 'props': [('background-color', '#ff3c00'), 
                                             ('color', 'white'),
                                             ('font-family', 'Orbitron, sans-serif')]},
                {'selector': 'td', 'props': [('color', '#00ffea'),
                                             ('font-family', 'Orbitron, sans-serif')]}
            ]
        )
        st.dataframe(styled_df, height=400)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;'>Made with 💫 by Your Team</p>",
    unsafe_allow_html=True
)
