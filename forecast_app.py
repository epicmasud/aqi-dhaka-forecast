import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import time  # লোডিং অ্যানিমেশনের জন্য

# --- Health advice function (এটা শুরুতে রাখা হয়েছে) ---
def get_health_advice(aqi):
    if aqi <= 50:
        return "বাতাস ভালো — বাইরে যাওয়া সম্পূর্ণ নিরাপদ।"
    elif aqi <= 100:
        return "মাঝারি দূষণ — সংবেদনশীল হলে সতর্ক থাকুন।"
    elif aqi <= 150:
        return "অস্বাস্থ্যকর সংবেদনশীলদের জন্য — বাইরে কম যান।"
    elif aqi <= 200:
        return "অস্বাস্থ্যকর — N95 মাস্ক পরুন, শারীরিক পরিশ্রম কমান।"
    elif aqi <= 300:
        return "খুব অস্বাস্থ্যকর — বাইরে না যাওয়াই ভালো।"
    else:
        return "বিপজ্জনক — জরুরি অবস্থা, ঘরে থাকুন, মাস্ক পরুন।"

# --- পেজ কনফিগ (প্রফেশনাল লুক) ---
st.set_page_config(
    page_title="Dhaka AQI Forecaster",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- কাস্টম CSS (প্রফেশনাল ডিজাইন) ---
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #00D4FF;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    .sub-header {
        font-size: 1.3rem;
        color: #B0BEC5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .aqi-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        background: linear-gradient(135deg, rgba(30, 40, 60, 0.9), rgba(10, 20, 40, 0.9));
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    .aqi-number {
        font-size: 4.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .aqi-level {
        font-size: 1.8rem;
        margin: 0.5rem 0;
    }
    .advice {
        font-size: 1.2rem;
        margin-top: 1.5rem;
    }
    .footer {
        text-align: center;
        color: #78909C;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #444;
    }
    </style>
""", unsafe_allow_html=True)

# --- হেডার ---
st.markdown('<h1 class="main-header">Dhaka AQI Forecaster</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time Air Quality Prediction for Dhaka using LSTM</p>', unsafe_allow_html=True)

# --- সাইডবার (প্রফেশনাল লুক) ---
with st.sidebar:
    st.title("Dhaka AQI")
    st.image("https://img.icons8.com/fluency/96/000000/air-quality.png", width=100)  # আইকন বা তোমার লোগো
    st.markdown("### Controls")
    st.info("Enter weather conditions to predict AQI")
    st.markdown("---")
    st.markdown("**Features**")
    st.markdown("- LSTM-based prediction")
    st.markdown("- Health advice")
    st.markdown("- Dark modern theme")
    st.markdown("---")
    st.markdown("**Developer**")
    st.markdown("Masud Hasan")
    st.markdown("[GitHub](https://github.com/epicmasud/aqi-dhaka-forecast)")
    st.markdown("[Feedback?](https://your-contact-link)")

# --- মেইন কনটেন্ট ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Parameters")
    temp = st.slider("Expected Average Temperature (°C)", 10.0, 40.0, 28.0, step=0.5)
    rain = st.slider("Expected Rainfall (mm)", 0.0, 100.0, 0.0, step=1.0)

    predict_button = st.button("Predict AQI", type="primary", use_container_width=True)

if predict_button:
    with st.spinner("Analyzing weather data and predicting AQI..."):
        time.sleep(1.5)  # সিমুলেট লোডিং (অপশনাল)
        try:
            # মডেল লোড
            model = load_model('dhaka_aqi_lstm.h5', custom_objects={'mse': MeanSquaredError()})

            # তোমার মডেলের আসল input shape অনুযায়ী
            num_features = 19  # ← এখানে তোমার model.input_shape[2] দাও
            timesteps = 7

            dummy_features = np.zeros((1, timesteps, num_features), dtype=np.float32)
            dummy_features[0, 0, 0] = temp
            dummy_features[0, 0, 1] = rain
            # অন্য ফিচার যোগ করো যদি থাকে (যেমন lag1, season)

            pred_scaled = model.predict(dummy_features, verbose=0)[0][0]
            pred_aqi = int(pred_scaled * 350 + 30)  # আসল scaler দিয়ে বদলাও

            # AQI কার্ড (প্রফেশনাল লুক)
            if pred_aqi <= 50:
                color, level, icon = "#4CAF50", "ভালো", "😊"
            elif pred_aqi <= 100:
                color, level, icon = "#FFEB3B", "মাঝারি", "😐"
            elif pred_aqi <= 150:
                color, level, icon = "#FF9800", "অস্বাস্থ্যকর", "😷"
            elif pred_aqi <= 200:
                color, level, icon = "#F44336", "অস্বাস্থ্যকর", "🚨"
            elif pred_aqi <= 300:
                color, level, icon = "#9C27B0", "খুব অস্বাস্থ্যকর", "☠️"
            else:
                color, level, icon = "#B71C1C", "বিপজ্জনক", "☢️"

            st.markdown(
                f"""
                <div class="aqi-card">
                    <div class="aqi-number" style="color:{color};">{pred_aqi}</div>
                    <div class="aqi-level" style="color:{color};">{level} {icon}</div>
                    <div class="advice">
                        {get_health_advice(pred_aqi)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"সমস্যা: {str(e)}\nমডেল ফাইল বা লাইব্রেরি চেক করুন।")

# --- ফুটার ---
st.markdown(
    """
    <div class="footer">
        © 2026 Dhaka AQI Forecaster | Built with ❤️ by Masud Hasan | 
        <a href="https://github.com/epicmasud/aqi-dhaka-forecast" style="color:#00D4FF;">Source Code</a>
    </div>
    """,
    unsafe_allow_html=True
)
