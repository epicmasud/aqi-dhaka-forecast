import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError  # mse-এর জন্য import এখানে করা হলো (try ব্লকের বাইরে)

st.title("Dhaka AQI Forecaster (LSTM Model)")
st.write("ঢাকার আগামী AQI প্রেডিক্ট করুন")

# ইউজার ইনপুট
temp = st.number_input("Expected Average Temperature (°C)", 10.0, 40.0, 28.0)
rain = st.number_input("Expected Rainfall (mm)", 0.0, 100.0, 0.0)

# ডামি ফিচার — তোমার মডেলের আসল input shape অনুযায়ী অ্যাডজাস্ট করো
# ধরে নিচ্ছি ৭ টাইমস্টেপ × ৬ ফিচার (যদি ভিন্ন হয় তাহলে model.summary() দেখে বদলাও)
dummy_features = np.array([[temp, rain, 0, 0, 0, 0]] * 7, dtype=np.float32)
dummy_features = dummy_features.reshape(1, 7, 6)  # shape: (1, timesteps, features)

if st.button("Predict AQI"):
    try:
        # মডেল লোড করো (custom_objects দিয়ে mse ফিক্স)
        model = load_model('dhaka_aqi_lstm.h5', custom_objects={'mse': MeanSquaredError()})

        # প্রেডিক্ট করো
        pred_scaled = model.predict(dummy_features, verbose=0)[0][0]

        # স্কেল ব্যাক (তোমার ট্রেইনিং-এর মিন-ম্যাক্স ব্যবহার করো)
        # এখানে ডামি স্কেলিং — আসল মিন/ম্যাক্স দিয়ে বদলাও
        pred_aqi = pred_scaled * 350 + 30

        st.success(f"**প্রেডিক্টেড AQI: {int(pred_aqi)}**")

        if pred_aqi > 150:
            st.warning("**স্বাস্থ্য পরামর্শ**: বাইরে N95 মাস্ক পরুন, শারীরিক পরিশ্রম কমান, সংবেদনশীল মানুষ ঘরে থাকুন।")
        elif pred_aqi > 100:
            st.info("**মাঝারি দূষণ** — সংবেদনশীল হলে বাইরের সময় কমান।")
        else:
            st.success("**ভালো বায়ুর গুণগত মান** — বাইরে যাওয়া নিরাপদ!")

    except Exception as e:
        st.error(f"সমস্যা: {str(e)}\nমডেল ফাইল 'dhaka_aqi_lstm.h5' আছে কি না চেক করুন বা requirements.txt-এ tensorflow==2.15.0 আছে কি না দেখুন।")
