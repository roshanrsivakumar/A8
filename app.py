# File: app.py

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.title("Customer Churn Predictor")
st.markdown("""
Developed by: Roshan Rajkumar Sivakumar  
Registration Number: V01151141
""")


# Load model
@st.cache_resource
def load_model():
    with open("simple_churn_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data['model'], data['features'], data['encoders']


model, model_features, label_encoders = load_model()


# Preprocessing
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    for col in df.columns:
        if df[col].dtype == "object":
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
            else:
                df[col] = 0
    return df[model_features].fillna(0)


mode = st.sidebar.selectbox("Select Mode", ["Single Customer", "Batch Prediction"])

# ------- Single Prediction -------
if mode == "Single Customer":
    st.subheader("Single Customer Input")
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.number_input("Tenure (months)", 0, 100, 12)
    charges = st.number_input("Monthly Charges", 0.0, 1000.0, 70.0)

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([{
                'gender': gender,
                'tenure': tenure,
                'MonthlyCharges': charges
            }])
            X = preprocess(input_df)
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0][1]
            st.success(f"Prediction: {'Churn' if pred else 'No Churn'} | Risk: {prob:.2%}")

            st.plotly_chart(
                px.bar(
                    x=["No Churn", "Churn"],
                    y=[1 - prob, prob],
                    labels={"x": "Outcome", "y": "Probability"},
                    color=["No Churn", "Churn"]
                )
            )
        except Exception as e:
            st.error(f"Error: {e}")

# ------- Batch Prediction -------
elif mode == "Batch Prediction":
    st.subheader("Upload Customer List for Batch Prediction")
    file = st.file_uploader("Upload CSV file", type="csv")
    if file:
        try:
            df = pd.read_csv(file)
            st.write("Input Preview", df.head())
            X = preprocess(df)
            df['Prediction'] = model.predict(X)
            df['Probability'] = model.predict_proba(X)[:, 1].round(2)
            df['Label'] = df['Prediction'].map({1: "Churn", 0: "No Churn"})
            st.subheader("Prediction Results")
            st.write(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("Download Predictions", csv, "churn_predictions.csv", "text/csv")

            # Charts
            st.subheader("Churn Distribution")
            counts = df['Label'].value_counts().reset_index()
            counts.columns = ['Churn Status', 'Count']
            st.plotly_chart(px.bar(counts, x='Churn Status', y='Count', color='Churn Status'))

            if 'tenure' in df.columns:
                st.subheader("Tenure by Churn")
                st.plotly_chart(px.histogram(df, x='tenure', color='Label'))

            if 'MonthlyCharges' in df.columns:
                st.subheader("Monthly Charges by Churn")
                st.plotly_chart(px.histogram(df, x='MonthlyCharges', color='Label'))

            if 'gender' in df.columns:
                st.subheader("Churn by Gender")
                fig = px.bar(
                    df.groupby(['gender', 'Label']).size().reset_index(name='count'),
                    x='gender', y='count', color='Label', barmode='group'
                )
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
