import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError  # mse-এর জন্য

st.title("Dhaka AQI Forecaster (LSTM Model)")
st.write("ঢাকার আগামী AQI প্রেডিক্ট করুন")

# ইউজার ইনপুট
temp = st.number_input("Expected Average Temperature (°C)", 10.0, 40.0, 28.0)
rain = st.number_input("Expected Rainfall (mm)", 0.0, 100.0, 0.0)

# ডামি ফিচার — তোমার মডেলের আসল input shape অনুযায়ী অ্যাডজাস্ট করো
# AQI2.ipynb-এ model.input_shape দেখে নাও (যেমন (None, 7, 6) হলে 7 timesteps × 6 features)
dummy_features = np.array([[temp, rain, 0, 0, 0, 0]] * 7, dtype=np.float32)
dummy_features = dummy_features.reshape(1, 7, 6)  # shape: (1, timesteps, features)

if st.button("Predict AQI"):
    try:
        model = load_model('dhaka_aqi_lstm.h5', custom_objects={'mse': MeanSquaredError()})

        num_features = 19  # ← এখানে তোমার model.input_shape[2] দাও
        dummy_features = np.zeros((1, 7, num_features), dtype=np.float32)
        dummy_features[0, 0, 0] = temp
        dummy_features[0, 0, 1] = rain
        # বাকি ফিচারগুলো ডামি রাখো বা ইউজার থেকে নাও

        pred_scaled = model.predict(dummy_features, verbose=0)[0][0]
        pred_aqi = pred_scaled * 350 + 30

        st.success(f"**প্রেডিক্টেড AQI: {int(pred_aqi)}**")
        # অ্যাডভাইস কোড...

    except Exception as e:
        st.error(f"সমস্যা: {str(e)}")
