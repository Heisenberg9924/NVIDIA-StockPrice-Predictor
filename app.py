import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import time


st.set_page_config(page_title="Stock Price Predictor", layout="wide")

from PIL import Image

col1, col2 = st.columns([6, 15])
with col1:
    logo = Image.open("nvidia-logo-vert.png")
    st.image(logo, width=200)
with col2:
    st.markdown("<h1 style='padding-top: 15px;'> NVIDIA Stock Price Predictor</h1>", unsafe_allow_html=True)

model=joblib.load("regression_model.pkl")
scale_feat=joblib.load("feature_scaler.pkl")
scale_tar=joblib.load("target_scaler.pkl")

df = yf.download("NVDA", period="11d", interval="1d", group_by='ticker')
df.columns = df.columns.droplevel(0)
close_vals = df['Close'].values.flatten()
if len(close_vals) >= 10:
    MA5 = pd.Series(close_vals[-5:]).mean()
    MA10 = pd.Series(close_vals[-10:]).mean()
    Return1 = (close_vals[-1] - close_vals[-2]) / close_vals[-2]
else:
    st.error("Insufficient data for MA5, MA10 calculation. Try again later.")
    st.stop()

today = df.iloc[-1]
yesterday = df.iloc[-2]
yesterday_close = yesterday['Close']

st.markdown("### ✔️ Today's Data (Auto-Fetched)")
st.markdown(f"""
- **Open:** {today['Open']}  
- **Close:** {today['Close']}  
- **Previous Close:** {yesterday_close}  
- **Volume:** {today['Volume']}  
- **MA5:** {MA5}  
- **MA10:** {MA10}  
""")
with st.form("input_form"):
    Open = st.slider(
        "Open", 
        min_value=float(today['Open']) * 0.9, 
        max_value=float(today['Open']) * 1.1, 
        value=float(today['Open']), 
        step=0.01
    )
    High = st.slider(
        "High", 
        min_value=float(today['High']) * 0.9, 
        max_value=float(today['High']) * 1.1, 
        value=float(today['High']), 
        step=0.01
    )
    Low = st.slider(
        "Low", 
        min_value=float(today['Low']) * 0.9, 
        max_value=float(today['Low']) * 1.1, 
        value=float(today['Low']), 
        step=0.01
    )
    Volume = st.slider(
        "Volume", 
        min_value=int(today['Volume'] * 0.8), 
        max_value=int(today['Volume'] * 1.2), 
        value=int(today['Volume']),
        step=10000
    )
    Close = st.slider(
        "Close", 
        min_value=float(today['Close']) * 0.9, 
        max_value=float(today['Close']) * 1.1, 
        value=float(today['Close']), 
        step=0.01
    )
    Close_lag1 = st.slider(
        "Previous Close", 
        min_value=float(yesterday['Close']) * 0.9, 
        max_value=float(yesterday['Close']) * 1.1, 
        value=float(yesterday['Close']),
        step=0.01
    )
    
   
    submitted = st.form_submit_button("Predict Tomorrow's Price")

if submitted:
    features = np.array([[Close, High, Low, Open, Volume, Close_lag1, MA5, MA10, Return1]])
    scaled_input = scale_feat.transform(features)
    scaled_output = model.predict(scaled_input)
    predicted_price = scale_tar.inverse_transform([[scaled_output[0]]])[0][0]
    st.success(f"📈 Predicted Closing Price Tomorrow: ${predicted_price:.2f}")

st.subheader("📈 Live Stock Price Chart (NVIDIA)")

placeholder = st.empty()
refresh_seconds = 30

with st.spinner("Loading..."):
    df = yf.download(tickers="NVDA", period="30d", interval="1h", progress=False)
    st.line_chart(df["Close"])

    df["MA5"]=df["Close"].rolling(window=5).mean()
    df["MA10"]=df["Close"].rolling(window=10).mean()

    col1,col2=st.columns(2)

    with col1:
        st.subheader("📊 MA5")
        st.line_chart(df["MA5"])
    with col2:
        st.subheader("📊 MA10")
        st.line_chart(df["MA10"])






