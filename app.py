import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Load Model
# -----------------------------
artifacts = joblib.load("ipl_decision_tree_pipeline.pkl")

model = artifacts["model"]
encoders = artifacts["encoders"]
features = artifacts["features"]

st.set_page_config(page_title="IPL Match Winner Prediction", page_icon="🏏")

st.title("🏏 IPL Match Winner Prediction")
st.write("Predict whether Team 1 will win the match.")
  
# -----------------------------
# Dropdown Values
# -----------------------------
team1 = st.selectbox("Team 1", encoders["team1"].classes_)
team2 = st.selectbox("Team 2", encoders["team2"].classes_)

venue = st.selectbox("Venue", encoders["venue"].classes_)
city = st.selectbox("City", encoders["city"].classes_)

toss_decision = st.selectbox(
    "Toss Decision",
    encoders["toss_decision"].classes_
)

team1_toss_winner = st.selectbox(
    "Did Team 1 Win Toss?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_df = pd.DataFrame({
        "team1": [team1],
        "team2": [team2],
        "venue": [venue],
        "city": [city],
        "toss_decision": [toss_decision],
        "team1_toss_winner": [team1_toss_winner]
    })

    # Encode categorical columns
    for col in ["team1", "team2", "venue", "city", "toss_decision"]:
        input_df[col] = encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success(f"🏆 Prediction: {team1} is likely to WIN!")
    else:
        st.success(f"🏆 Prediction: {team2} is likely to WIN!")