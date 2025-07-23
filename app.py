import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model and data
df = pickle.load(open('df.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title("💻 Laptop Price Predictor")

# Input fields
Company = st.selectbox("Brand", df['Company'].unique())
TypeName = st.selectbox("Type", df['TypeName'].unique())
Ram = st.selectbox("Ram (in GB)", [2, 4, 6, 8, 12, 16, 32])
Weight = st.number_input("Weight (in KG)", min_value=0.0, max_value=5.0, step=0.1)
Touchscreen = st.selectbox("Touchscreen", ['No', 'Yes'])
Ips = st.selectbox("IPS", ['No', 'Yes'])
ScreenSize = st.number_input('Screen Size (inches)', min_value=10.0, max_value=18.0, step=0.1)
Resolution = st.selectbox("Screen Resolution", ['1920x1080','1366x768','1600x900', '3840x2160',
                                                 '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
Cpu = st.selectbox('CPU Brand', df['Cpubrand'].unique())
Hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
Ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
Gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
Os = st.selectbox('Operating System', df['os'].unique())  # ensure column is 'os' in df

if st.button("💰 Predict Price"):

    # Encode categorical values
    Touchscreen = 1 if Touchscreen == 'Yes' else 0
    Ips = 1 if Ips == 'Yes' else 0

    # Calculate PPI
    X_res = int(Resolution.split('x')[0])
    Y_res = int(Resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2)**0.5) / ScreenSize

    # Create input array
    query = np.array([Company, TypeName, Ram, Weight, Touchscreen, Ips, ppi, Cpu, Hdd, Ssd, Gpu, Os])
    query = query.reshape(1, 12)

    # Predict and display
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.subheader(f"💵 Estimated Laptop Price: ₹ {predicted_price}")
