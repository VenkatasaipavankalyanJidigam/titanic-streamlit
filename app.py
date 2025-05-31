
import streamlit as st
import numpy as np
import joblib

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title(" Titanic Survival Predictor")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

sex = 1 if sex == "male" else 0
embarked_map = {'C': 0, 'Q': 1, 'S': 2}
embarked = embarked_map[embarked]

features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
features_scaled = scaler.transform(features)

if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]
    if prediction == 1:
        st.success(f"Survived! (Probability: {prob:.2f})")
    else:
        st.error(f"Did not survive (Probability: {prob:.2f})")
