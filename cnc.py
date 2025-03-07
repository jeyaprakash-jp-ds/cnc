import streamlit as st
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf
import pickle
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ReLU

def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)
encoder=load_model("Encoder_MP.pkl")
scaler=load_model("scaler.pkl")
model = tf.keras.models.load_model("final.h5")

st.write(**"FINAL CNC PREDICTION PROJECT" **)
uploaded_file = st.file_uploader("Upload an Excel file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, dtype=str)
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
        if 'Machining_Process' in df.columns:
            df['Machining_Process'] = encoder.fit_transform(df[["Machining_Process"]])
        else:
            st.error("Column 'Machining_Process' not found in the uploaded file.")
            
    except Exception as e:
        st.error(f"Error: {e}")
    st.write(df)
    if st.button('Test Result'):
        df_scaled = scaler.fit_transform(df)
        prediction = model.predict(df_scaled)
        st.subheader("Predicted Test Result")
        df["Tool Wear"] = ["Worn" if p[0] > 0.5 else "Unworn" for p in prediction]
        df["Visual inspection"] = ["Properly Clamped" if p[1] > 0.5 else "Not Properly Clamped" for p in prediction]
        df["Machining Completion"] = ["Completed" if p[2] > 0.5 else "Not Completed" for p in prediction]
        st.write(df)
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")
