# ===== Importing Required Libraries =====
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ===== API Key for OpenWeatherMap =====
API_KEY = "d14a4f432f95fbcc237c73076e774343"

# ===== Page Setup =====
st.set_page_config("🌤️ Air Quality & Weather Advisor", layout="wide")
st.title("🌍 Smart Air Quality & Weather Assistant")

# ======= LSTM Helper Functions =======
def prepare_data(df, steps=3):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(len(df_scaled) - steps):
        X.append(df_scaled[i:i+steps])
        y.append(df_scaled[i+steps])
    return np.array(X), np.array(y), scaler

@st.cache_resource
def train_lstm_model(past_df):
    X, y, scaler = prepare_data(past_df[['pm2_5', 'pm10', 'so2', 'no2']])
    if len(X) == 0:
        return None, None
    
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)
    return model, scaler

def predict_future(model, scaler, past_df, steps=4):
    data = scaler.transform(past_df[['pm2_5', 'pm10', 'so2', 'no2']])
    predictions = []
    input_seq = data[-3:].copy()
    for _ in range(steps):
        input_seq_reshaped = np.expand_dims(input_seq, axis=0)
        pred = model.predict(input_seq_reshaped, verbose=0)[0]
        predictions.append(pred)
        input_seq = np.vstack([input_seq[1:], pred])
    predictions = scaler.inverse_transform(predictions)
    dates = pd.date_range(datetime.now(), periods=steps).date
    return pd.DataFrame(predictions, columns=["pm2_5", "pm10", "so2", "no2"], index=dates)

# ======= API Functions =======
def get_coordinates(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
        res = requests.get(url).json()
        return (res['coord']['lat'], res['coord']['lon']) if 'coord' in res else (None, None)
    except:
        return None, None

def get_current_weather(lat, lon):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={API_KEY}"
        res = requests.get(url).json()
        return {
            "temp": res['main']['temp'],
            "humidity": res['main']['humidity'],
            "wind_speed": res['wind']['speed'],
            "wind_deg": res['wind'].get('deg', 0)
        }
    except:
        return None

def get_air_quality(lat, lon):
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
        res = requests.get(url).json()
        return pd.DataFrame([{
            "datetime": pd.to_datetime(i['dt'], unit='s'),
            "pm2_5": i['components']['pm2_5'],
            "pm10": i['components']['pm10'],
            "so2": i['components']['so2'],
            "no2": i['components']['no2']
        } for i in res.get('list', [])])
    except:
        return pd.DataFrame()

def deg_to_direction(deg):
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return dirs[round(deg / 45) % 8]

# ======= Health Advice =======
def get_suggestions(condition, pm2_5):
    if pm2_5 <= 12:
        status = "Good"
    elif pm2_5 <= 35:
        status = "Moderate"
    elif pm2_5 <= 55:
        status = "Unhealthy for Sensitive Groups"
    elif pm2_5 <= 150:
        status = "Unhealthy"
    else:
        status = "Very Unhealthy"
    
    recs = {
        "Asthma": "Carry inhaler, avoid exertion, wear a mask.",
        "Heart Disease": "Avoid exercise, stay indoors, wear a mask.",
        "Children": "Keep indoors, avoid outdoor play.",
        "Elderly": "Stay hydrated and indoors, wear a mask.",
        "Healthy": "Wear a mask on poor AQI days."
    }
    return status, recs.get(condition, "Avoid pollution exposure.")

# ======= Interactive Chatbot =======
def chatbot_response(user_msg, condition, pm2_5, city):
    user_msg = user_msg.lower()

    # Air quality status
    if pm2_5 <= 12:
        level, emoji = "Good", "✅"
    elif pm2_5 <= 35:
        level, emoji = "Moderate", "⚠️"
    elif pm2_5 <= 55:
        level, emoji = "Unhealthy (Sensitive)", "🚩"
    else:
        level, emoji = "Unhealthy", "🚫"

    recs = {
        "asthma": "Carry your inhaler and avoid outdoor exposure.",
        "heart disease": "Stay indoors, avoid exertion.",
        "children": "Indoor play is best today.",
        "elderly": "Stay hydrated and limit outdoor movement.",
        "healthy": "Outdoor activity okay but avoid pollution-heavy areas."
    }

    if "hi" in user_msg or "hello" in user_msg:
        return f"👋 Hi there! Air quality in **{city}** is currently **{level} {emoji}**.\nHow can I assist you today?"

    elif "can i go" in user_msg or "safe" in user_msg or "outside" in user_msg:
        if pm2_5 <= 35:
            return f"✅ Yes, it's safe to go outside in **{city}**. Air quality is **{level}**.\nAdvice for {condition}: {recs[condition.lower()]}"
        else:
            return f"🚫 Air is **{level}** in **{city}**. Best to limit outdoor exposure.\nAdvice for {condition}: {recs[condition.lower()]}"

    elif "precaution" in user_msg or "what should i do" in user_msg or "mask" in user_msg:
        return f"😷 For {condition}: {recs[condition.lower()]}\nCurrent AQI: **{level} {emoji}**."

    else:
        return f"🤖 I can help with air safety and precautions.\nTry asking: 'Is it safe to go outside?' or 'What should I do?'"

# ======= Streamlit UI =======
city = st.text_input("🏙️ Enter a city name:")
health_condition = st.selectbox("Select your health condition:", ["Healthy", "Asthma", "Heart Disease", "Children", "Elderly"])

if city:
    lat, lon = get_coordinates(city)
    if lat:
        st.subheader("🌡️ Current Weather & 🗺️ City Map")
        col1, col2 = st.columns(2)

        with col1:
            weather = get_current_weather(lat, lon)
            if weather:
                st.metric("Temperature (°C)", f"{weather['temp']:.1f}")
                st.metric("Humidity (%)", f"{weather['humidity']}")
                st.metric("Wind Speed (m/s)", f"{weather['wind_speed']:.1f}")
                st.metric("Wind Direction", deg_to_direction(weather["wind_deg"]))

        with col2:
            m = folium.Map(location=[lat, lon], zoom_start=11)
            folium.Marker([lat, lon], tooltip=city).add_to(m)
            folium_static(m)

        st.subheader("📊 AQI Forecast")
        aqi_df = get_air_quality(lat, lon)
        
        if not aqi_df.empty:
            aqi_df = aqi_df.set_index("datetime").resample("D").mean().reset_index()
            past_df = aqi_df.tail(7).copy()

            # Train model and predict
            model, scaler = train_lstm_model(past_df)
            
            if model is not None and scaler is not None:
                future_df = predict_future(model, scaler, past_df)

                full_df = pd.concat([past_df.set_index("datetime")[["pm2_5", "pm10", "so2", "no2"]],
                                     future_df.rename_axis("datetime")])
                fig = px.line(full_df, x=full_df.index, y=full_df.columns, title="Predicted AQI (μg/m³)")
                st.plotly_chart(fig, use_container_width=True)

                latest_pm2_5 = future_df.iloc[0]['pm2_5']
                status, message = get_suggestions(health_condition, latest_pm2_5)
                st.success(f"**Predicted Air Quality:** {status}\n\n**Advice for {health_condition}:** {message}")
            else:
                st.warning("Insufficient data for prediction. Showing current data only.")
                fig = px.line(past_df, x="datetime", y=["pm2_5", "pm10", "so2", "no2"], 
                             title="Current AQI Data (μg/m³)")
                st.plotly_chart(fig, use_container_width=True)
                
                latest_pm2_5 = past_df.iloc[-1]['pm2_5']
                status, message = get_suggestions(health_condition, latest_pm2_5)
                st.info(f"**Current Air Quality:** {status}\n\n**Advice for {health_condition}:** {message}")

            # ======= CHATBOT SECTION (COMPLETION) =======
            st.subheader("🤖 Chatbot Assistant")
            user_msg = st.text_input("💬 Ask something about air safety (e.g., 'Can I go outside?')")
            
            if user_msg:
                # Get latest PM2.5 value for chatbot
                current_pm2_5 = latest_pm2_5 if 'latest_pm2_5' in locals() else 25
                
                # Generate and display chatbot response
                bot_response = chatbot_response(user_msg, health_condition, current_pm2_5, city)
                
                # Display the response in a chat-like format
                with st.chat_message("assistant"):
                    st.markdown(bot_response)
                
                # Optional: Add to chat history using session state
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append({
                    "user": user_msg,
                    "assistant": bot_response
                })
                
                # Show recent chat history
                if len(st.session_state.chat_history) > 1:
                    with st.expander("💬 Recent Chat History"):
                        for i, chat in enumerate(st.session_state.chat_history[-3:]):  # Show last 3
                            st.write(f"**You:** {chat['user']}")
                            st.write(f"**Assistant:** {chat['assistant']}")
                            st.write("---")
        else:
            st.error("Unable to fetch air quality data. Please check your internet connection.")
    else:
        st.error("City not found. Please check the spelling and try again.")
else:
    st.info("👆 Enter a city name to get started!")

                    st.markdown(f"<div class='metric-container'>📊 Mean Absolute Error (MAE): {round(mae, 2)}</div>", unsafe_allow_html=True)
                    future_fig = go.Figure()
                    future_fig.add_trace(go.Scatter(x=future_times, y=future_pm2_5, mode='lines+markers', name='Predicted PM2.5'))
                    future_fig.update_layout(title="📈 Predicted PM2.5 for Next Hours", xaxis_title="Time", yaxis_title="PM2.5 Level")
                    st.plotly_chart(future_fig, use_container_width=True)
