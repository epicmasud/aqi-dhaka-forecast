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
        # মডেল লোড করো (custom_objects দিয়ে mse ডেসিরিয়ালাইজ ফিক্স)
        model = load_model('dhaka_aqi_lstm.h5', custom_objects={'mse': MeanSquaredError()})

        # তোমার মডেলের আসল input shape অনুযায়ী ফিচার সংখ্যা
        # AQI2.ipynb-এ রান করো: print(model.input_shape) → (None, timesteps, num_features)
        # উদাহরণ: যদি (None, 7, 19) হয় তাহলে num_features = 19
        num_features = 19  # ← এখানে তোমার আসল নাম্বার দাও
        timesteps = 7      # সাধারণত 7

        # ডামি ফিচার তৈরি করো (সব ০ রাখো, শুধু প্রথম টাইমস্টেপে ইউজার ইনপুট দাও)
        dummy_features = np.zeros((1, timesteps, num_features), dtype=np.float32)
        dummy_features[0, 0, 0] = temp   # প্রথম ফিচারে তাপমাত্রা
        dummy_features[0, 0, 1] = rain   # দ্বিতীয় ফিচারে বৃষ্টি
        # বাকি ফিচারগুলো ডামি রাখা (যদি তোমার মডেলে অন্য ফিচার থাকে → এখানে যোগ করো)

        # প্রেডিক্ট করো
        pred_scaled = model.predict(dummy_features, verbose=0)[0][0]

        # স্কেল ব্যাক (তোমার ট্রেইনিং-এর scaler.data_min_ / data_max_ ব্যবহার করো)
        # এখানে ডামি — আসল মিন/ম্যাক্স দিয়ে বদলাও
        pred_aqi = pred_scaled * 350 + 30

        # প্রেডিকশন দেখাও
        st.success(f"**প্রেডিক্টেড AQI: {int(pred_aqi)}**")

        # AQI ক্যাটাগরি + অ্যাডভাইস
        if pred_aqi <= 50:
            st.success("**ভালো বায়ু** — বাইরে যাওয়া নিরাপদ!")
        elif pred_aqi <= 100:
            st.info("**মাঝারি** — সংবেদনশীল হলে সতর্ক থাকুন।")
        elif pred_aqi <= 150:
            st.warning("**অস্বাস্থ্যকর সংবেদনশীলদের জন্য** — বাইরে কম যান।")
        elif pred_aqi <= 200:
            st.error("**অস্বাস্থ্যকর** — N95 মাস্ক পরুন, পরিশ্রম কমান।")
        elif pred_aqi <= 300:
            st.error("**খুব অস্বাস্থ্যকর** — বাইরে না যাওয়াই ভালো।")
        else:
            st.error("**বিপজ্জনক** — জরুরি অবস্থা, ঘরে থাকুন।")

    except Exception as e:
        st.error(f"সমস্যা: {str(e)}\nমডেল ফাইল 'dhaka_aqi_lstm.h5' আছে কি না চেক করুন বা requirements.txt-এ tensorflow আছে কি না দেখুন।")
