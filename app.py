import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
df = pickle.load(open('df.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Create input fields for user input
Company = st.selectbox("Brand", df['Company'].unique())
TypeName = st.selectbox("Type", df['TypeName'].unique())
Ram = st.selectbox("Ram (in GB)", [2, 4, 6, 8, 12, 16, 32])
Weight = st.number_input("Weight (in KG)", min_value=0.0, max_value=5.0, step=0.1)
Touchscreen = st.selectbox("Touchscreen", ['No', 'Yes'])
Ips = st.selectbox("IPS", ['No', 'Yes'])
ScreenSize = st.number_input('Screen Size')
Resolution = st.selectbox("Resolution", ['1920x1080','1366x768','1600x900', '3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
Cpu = st.selectbox('Brand', df['Cpubrand'].unique())
Hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
dd = st.selectbox('SSD (in GB)', [0, 8,128, 256, 512, 1024])
Gpu = st.selectbox('GPU', df['Gpu brand'].unique())
Os = st.selectbox('OS', df['os'].unique())

if st.button("Predict Price"):
  
    PPI=None
    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:   
        Touchscreen = 0
   
    if Ips == 'Yes':
        Ips = 1
    else: 
        Ips = 0
        
        X_res= int(Resolution.split('x')['0'])
        Y_res = int(Resolution.split('x')['1'])
        ppi= ((int(X_res)**2 + int(Y_res)**2)**0.5)/ScreenSize
        
    query = np.array([Company,TypeName,Ram,Weight,Touchscreen,Ips,PPI,Cpu,Hdd,dd,Gpu,Os])
    query = query.reshape(1, 12)
    st.title(f"Estimated Price: {int(np.exp(pipe.predict(query)[0]))}")
