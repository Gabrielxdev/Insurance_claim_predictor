import streamlit as st
import pandas as pd
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("label_encoder_gender.pkl")
le_diabetic = joblib.load("label_encoder_diabetic.pkl")
le_smoker = joblib.load("label_encoder_smoker.pkl")
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Insurance Claim Predictor", layout="centered")
st.title("Health Insurance Payment Prediction App")
st.write("Enter the details below to estimate your insurance payment amount.")


with st.form("input form"): #Pegue um carrinho de compras (with st.form).
    col1, col2 = st.columns(2) #Dentro desse carrinho, crie duas divisórias (st.columns).
    with col1: #Coloque os campos de idade, IMC e filhos na primeira divisória (with col1).
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    with col2: #Coloque os campos de pressão, gênero, etc., na segunda divisória (with col2).
        bloodpressure = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
        gender = st.selectbox("Gender", options= le_gender.classes_)
        diabetic = st.selectbox("Diabetic", options= le_diabetic.classes_)
        smoker = st.selectbox("Smoker", options= le_smoker.classes_)
#Espere até eu clicar no botão "Ir para o Caixa" (st.form_submit_button) para processar tudo que está no carrinho.
    submitted = st.form_submit_button("Predict Payment") #é o caixa do supermercado. Somente quando você clica nele é que todos os "produtos" (os dados do formulário) são processados de uma só vez.

if submitted:
    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "bloodpressure": [bloodpressure],
        "gender": [gender],
        "diabetic": [diabetic],
        "smoker": [smoker]
    })

    input_data["gender"] = le_gender.transform(input_data["gender"])
    input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])
    input_data["smoker"] = le_smoker.transform(input_data["smoker"])    

    num_cols = ["age", "bmi", "bloodpressure", "children"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Reorder columns to match the model's expected feature order
    expected_order = ['age', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker', 'gender']
    input_data = input_data[expected_order]

    prediction = model.predict(input_data)[0]
    st.success(f"**Estimated Insurance Payment: ${prediction:,.2f}**")


