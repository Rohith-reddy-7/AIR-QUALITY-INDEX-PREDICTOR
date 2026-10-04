<<<<<<< HEAD
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

=======
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


class MinMaxScaler:
    def fit_transform(self, values):
        self.minimum = np.min(values, axis=0)
        self.maximum = np.max(values, axis=0)
        self.range = np.where(self.maximum == self.minimum, 1, self.maximum - self.minimum)
        return (values - self.minimum) / self.range

    def transform(self, values):
        return (values - self.minimum) / self.range

    def inverse_transform(self, values):
        return values * self.range + self.minimum


st.set_page_config(page_title="Air Quality Intelligence", page_icon="AQ", layout="wide")

POLLUTANT_COLUMNS = ["pm2_5", "pm10", "so2", "no2"]
POLLUTANT_LABELS = {"pm2_5": "PM2.5", "pm10": "PM10", "so2": "SO2", "no2": "NO2"}
DB_PATH = Path(__file__).with_name("air_readings.db")


def get_api_key():
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    if api_key:
        return api_key
    try:
        return st.secrets.get("OPENWEATHER_API_KEY", "")
    except Exception:
        return ""


API_KEY = get_api_key()


def get_secret(name):
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


GROK_API_KEY = get_secret("GROK_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-mini")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def request_json(endpoint, params):
    if not API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not configured")
    response = requests.get(endpoint, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def init_database():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS readings (
                city TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                observed_at TEXT NOT NULL,
                pm2_5 REAL NOT NULL,
                pm10 REAL NOT NULL,
                so2 REAL NOT NULL,
                no2 REAL NOT NULL,
                UNIQUE(city, observed_at)
            )
            """
        )


def save_reading(city, latitude, longitude, reading):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO readings
            (city, latitude, longitude, observed_at, pm2_5, pm10, so2, no2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                city,
                latitude,
                longitude,
                reading["datetime"].isoformat(),
                reading["pm2_5"],
                reading["pm10"],
                reading["so2"],
                reading["no2"],
            ),
        )


def load_readings(city, days=14):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with sqlite3.connect(DB_PATH) as connection:
        readings = pd.read_sql_query(
            """
            SELECT observed_at AS datetime, pm2_5, pm10, so2, no2
            FROM readings
            WHERE city = ? AND observed_at >= ?
            ORDER BY observed_at
            """,
            connection,
            params=(city, cutoff),
        )
    if not readings.empty:
        readings["datetime"] = pd.to_datetime(readings["datetime"], utc=True)
    return readings


@st.cache_data(ttl=600, show_spinner=False)
def get_coordinates(city):
    data = request_json(
        "https://api.openweathermap.org/data/2.5/weather",
        {"q": city, "appid": API_KEY},
    )
    return data["coord"]["lat"], data["coord"]["lon"], data["name"]


@st.cache_data(ttl=600, show_spinner=False)
def get_current_weather(latitude, longitude):
    data = request_json(
        "https://api.openweathermap.org/data/2.5/weather",
        {"lat": latitude, "lon": longitude, "units": "metric", "appid": API_KEY},
    )
    return {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data.get("wind", {}).get("speed", 0),
        "wind_deg": data.get("wind", {}).get("deg", 0),
        "observed_at": pd.to_datetime(data["dt"], unit="s", utc=True),
    }


def parse_pollution_item(item):
    components = item["components"]
    return {
        "datetime": pd.to_datetime(item["dt"], unit="s", utc=True),
        "pm2_5": float(components.get("pm2_5", 0)),
        "pm10": float(components.get("pm10", 0)),
        "so2": float(components.get("so2", 0)),
        "no2": float(components.get("no2", 0)),
    }


@st.cache_data(ttl=600, show_spinner=False)
def get_air_quality(latitude, longitude):
    current_data = request_json(
        "https://api.openweathermap.org/data/2.5/air_pollution",
        {"lat": latitude, "lon": longitude, "appid": API_KEY},
    )
    forecast_data = request_json(
        "https://api.openweathermap.org/data/2.5/air_pollution/forecast",
        {"lat": latitude, "lon": longitude, "appid": API_KEY},
    )
    current = parse_pollution_item(current_data["list"][0])
    forecast = pd.DataFrame(
        [parse_pollution_item(item) for item in forecast_data.get("list", [])]
    )
    return current, forecast


def prepare_data(readings, steps=8):
    scaler = MinMaxScaler()
    values = scaler.fit_transform(readings[POLLUTANT_COLUMNS].to_numpy())
    features, targets = [], []
    for index in range(len(values) - steps):
        features.append(values[index : index + steps])
        targets.append(values[index + steps])
    return np.array(features), np.array(targets), scaler


@st.cache_resource(show_spinner=False)
def train_lstm_model(readings):
    if len(readings) < 25:
        return None, None
    try:
        from tensorflow.keras.layers import Dense, LSTM
        from tensorflow.keras.models import Sequential
    except ImportError:
        return None, None

    features, targets, scaler = prepare_data(readings)
    model = Sequential(
        [
            LSTM(32, activation="relu", input_shape=(features.shape[1], features.shape[2])),
            Dense(len(POLLUTANT_COLUMNS)),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(features, targets, epochs=15, batch_size=8, verbose=0)
    return model, scaler


def baseline_prediction(readings, steps=8):
    recent = readings[POLLUTANT_COLUMNS].tail(min(8, len(readings)))
    values = recent.mean().to_numpy() if not recent.empty else np.zeros(4)
    start = readings["datetime"].max() if not readings.empty else pd.Timestamp.now(tz="UTC")
    dates = pd.date_range(start + pd.Timedelta(hours=3), periods=steps, freq="3h")
    return pd.DataFrame(np.tile(values, (steps, 1)), columns=POLLUTANT_COLUMNS, index=dates)


def provider_forecast_prediction(forecast, steps):
    if forecast.empty:
        return pd.DataFrame(columns=POLLUTANT_COLUMNS)
    return (
        forecast.head(steps)
        .set_index("datetime")[POLLUTANT_COLUMNS]
        .clip(lower=0)
    )


def predict_future(model, scaler, readings, steps=8):
    if model is None or scaler is None:
        return baseline_prediction(readings, steps)
    values = scaler.transform(readings[POLLUTANT_COLUMNS].to_numpy())
    input_sequence = values[-8:].copy()
    predictions = []
    for _ in range(steps):
        prediction = model.predict(input_sequence[np.newaxis, :, :], verbose=0)[0]
        predictions.append(prediction)
        input_sequence = np.vstack([input_sequence[1:], prediction])
    start = readings["datetime"].max()
    dates = pd.date_range(start + pd.Timedelta(hours=3), periods=steps, freq="3h")
    predictions = np.maximum(scaler.inverse_transform(predictions), 0)
    return pd.DataFrame(predictions, columns=POLLUTANT_COLUMNS, index=dates)


def rolling_baseline_mae(readings, window=8):
    if len(readings) <= window:
        return None
    actual = readings[POLLUTANT_COLUMNS].to_numpy()[window:]
    predicted = np.array(
        [
            readings[POLLUTANT_COLUMNS].iloc[index - window : index].mean().to_numpy()
            for index in range(window, len(readings))
        ]
    )
    return float(np.abs(actual - predicted).mean())


def pm25_to_aqi(pm25):
    value = np.floor(max(0.0, min(float(pm25), 500.4)) * 10) / 10
    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 500.4, 301, 500),
    ]
    for low, high, aqi_low, aqi_high in breakpoints:
        if low <= value <= high:
            return round((aqi_high - aqi_low) / (high - low) * (value - low) + aqi_low)
    return 500


def aqi_label(aqi):
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def get_suggestions(condition, aqi):
    advice = {
        "Asthma": "Carry your inhaler and avoid strenuous outdoor activity.",
        "Heart Disease": "Avoid strenuous activity and consider staying indoors.",
        "Children": "Limit outdoor play and choose cleaner indoor activities.",
        "Elderly": "Limit prolonged outdoor exposure and stay hydrated.",
        "Healthy": "Outdoor activity is reasonable, but avoid heavily polluted areas.",
    }
    return aqi_label(aqi), advice.get(condition, "Limit pollution exposure.")


def deg_to_direction(degrees):
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return directions[round(degrees / 45) % 8]


def chatbot_response(user_message, condition, aqi, city):
    message = user_message.lower()
    level = aqi_label(aqi)
    _, advice = get_suggestions(condition, aqi)
    if any(word in message for word in ("hi", "hello")):
        return (
            f"Hi! I am your air-quality assistant for **{city}**. "
            f"The current AQI is **{aqi} ({level})**. "
            "You can ask whether it is safe to go outside, exercise, or what precautions to take."
        )
    if any(word in message for word in ("safe", "outside", "go out", "exercise")):
        if aqi <= 100:
            return f"Outdoor activity is generally reasonable in **{city}**. {advice}"
        return f"Air quality is **{level}** in **{city}**. Limit outdoor exposure. {advice}"
    if any(word in message for word in ("precaution", "should i", "mask", "health")):
        return f"For {condition}: {advice} Current AQI: **{aqi} ({level})**."
    if any(word in message for word in ("forecast", "tomorrow", "later", "next", "future")):
        return (
            f"The current AQI in **{city}** is **{aqi} ({level})**. "
            "Open the Prediction Lab to compare the live three-hour forecast, rolling baseline, "
            "and LSTM when enough observations are available."
        )
    if any(word in message for word in ("why", "analysis", "mean", "level", "bad", "good")):
        return (
            f"Analysis for **{city}**: AQI **{aqi}** is classified as **{level}**. "
            f"For the {condition} profile, the current guidance is: {advice} "
            "The recommendation is based on PM2.5 concentration and the selected health profile."
        )
    if any(word in message for word in ("pollution", "pollutant", "pm2", "pm10", "air")):
        return (
            f"The live analysis for **{city}** is AQI **{aqi} ({level})**. "
            f"Your profile is **{condition}**. {advice} "
            "Use Historical analytics to inspect individual pollutant trends."
        )
    return (
        f"I analyzed your question using the current **{city}** context: AQI **{aqi} ({level})**, "
        f"health profile **{condition}**. {advice} "
        "For a more detailed answer, ask about safety, exercise, health effects, pollutants, "
        "or the forecast."
    )


def grok_response(user_message, condition, aqi, city):
    if not GROK_API_KEY:
        return None
    level, advice = get_suggestions(condition, aqi)
    payload = {
        "model": GROK_MODEL,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an air-quality safety assistant. Explain the verified result "
                    "in plain language. Do not invent measurements, diagnose illness, "
                    "recommend medication, or change the provided AQI category or advice. "
                    "Keep the answer under 100 words and mention that it is not medical advice."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"City: {city}\nHealth profile: {condition}\nAQI: {aqi}\n"
                    f"AQI category: {level}\nVerified advice: {advice}\n"
                    f"Question: {user_message}"
                ),
            },
        ],
    }
    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, requests.RequestException, ValueError):
        return None


def gemini_response(user_message, condition, aqi, city):
    if not GEMINI_API_KEY:
        st.session_state.ai_status = "Gemini is not configured; using the local safety assistant."
        return None
    level, advice = get_suggestions(condition, aqi)
    prompt = (
        "You are an air-quality safety assistant. Explain the verified result in plain language. "
        "Do not invent measurements, diagnose illness, recommend medication, or change the "
        "provided AQI category or advice. Keep the answer under 100 words and mention that "
        "it is not medical advice.\n\n"
        f"City: {city}\nHealth profile: {condition}\nAQI: {aqi}\n"
        f"AQI category: {level}\nVerified advice: {advice}\nQuestion: {user_message}"
    )
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    for attempt in range(2):
        try:
            response = requests.post(
                endpoint,
                headers={"x-goog-api-key": GEMINI_API_KEY},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=20,
            )
            response.raise_for_status()
            answer = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            st.session_state.ai_status = "Gemini response active."
            return answer
        except requests.HTTPError as error:
            status_code = error.response.status_code if error.response is not None else 0
            if status_code in (429, 500, 502, 503, 504) and attempt == 0:
                time.sleep(1)
                continue
            st.session_state.ai_status = f"Gemini request failed ({status_code}); using the local safety assistant."
            return None
        except (KeyError, requests.RequestException, ValueError):
            st.session_state.ai_status = "Gemini response was unavailable; using the local safety assistant."
            return None


def ai_response(user_message, condition, aqi, city):
    return gemini_response(user_message, condition, aqi, city) or grok_response(
        user_message, condition, aqi, city
    )


def render_map(latitude, longitude, city):
    try:
        import folium
        from streamlit_folium import folium_static
    except ImportError:
        st.info("Install folium and streamlit-folium to display the city map.")
        return
    city_map = folium.Map(location=[latitude, longitude], zoom_start=11)
    folium.Marker([latitude, longitude], tooltip=city).add_to(city_map)
    folium_static(city_map, width=700, height=360)


def render_dashboard(city, health_condition, history_days, forecast_steps, selected_pollutants, alert_threshold, use_ai):
    if hasattr(st, "fragment"):
        st.caption("Live data refreshes automatically every 3 hours.")
    else:
        st.warning("Upgrade Streamlit to enable automatic three-hour refreshes.")
    if st.button("Refresh live data"):
        st.cache_data.clear()
        st.rerun()

    try:
        latitude, longitude, resolved_city = get_coordinates(city)
        weather = get_current_weather(latitude, longitude)
        current_reading, forecast = get_air_quality(latitude, longitude)
    except (KeyError, requests.RequestException, RuntimeError, ValueError) as error:
        st.error(f"Unable to retrieve live data: {error}")
        return

    save_reading(resolved_city, latitude, longitude, current_reading)
    history = load_readings(resolved_city, history_days)
    if history.empty:
        history = pd.DataFrame([current_reading])

    current_aqi = pm25_to_aqi(current_reading["pm2_5"])
    status, advice = get_suggestions(health_condition, current_aqi)
    forecast_peak = pm25_to_aqi(forecast["pm2_5"].max()) if not forecast.empty else current_aqi

    if current_aqi >= alert_threshold or forecast_peak >= alert_threshold:
        st.error(f"Air-quality alert: current AQI is {current_aqi}; forecast peak is {forecast_peak}.")
    else:
        st.success(f"Current air quality is {status}. Forecast peak AQI: {forecast_peak}.")

    st.subheader(f"{resolved_city}: live air-quality intelligence")
    metric_columns = st.columns(5)
    metric_columns[0].metric("PM2.5 AQI", current_aqi, status)
    metric_columns[1].metric("PM2.5", f"{current_reading['pm2_5']:.1f} ug/m3")
    metric_columns[2].metric("Temperature", f"{weather['temp']:.1f} C")
    metric_columns[3].metric("Humidity", f"{weather['humidity']}%")
    metric_columns[4].metric("Stored readings", len(history))
    st.info(f"Personalized guidance for {health_condition}: {advice}")

    overview_tab, analytics_tab, prediction_tab, assistant_tab = st.tabs(
        ["Overview", "Historical analytics", "Prediction lab", "Safety assistant"]
    )

    with overview_tab:
        weather_col, map_col = st.columns([1, 1.5])
        with weather_col:
            st.write(f"**Wind:** {weather['wind_speed']:.1f} m/s {deg_to_direction(weather['wind_deg'])}")
            st.write(f"**Last observed:** {current_reading['datetime'].strftime('%Y-%m-%d %H:%M UTC')}")
            if not forecast.empty:
                st.write("**Next forecast intervals**")
                forecast_view = forecast.head(6).copy()
                forecast_view["datetime"] = forecast_view["datetime"].dt.strftime("%d %b %H:%M")
                st.dataframe(forecast_view[["datetime", "pm2_5", "pm10"]], hide_index=True, use_container_width=True)
        with map_col:
            render_map(latitude, longitude, resolved_city)

    with analytics_tab:
        st.write("Observed readings collected by this application")
        observed = history.set_index("datetime")[selected_pollutants].tail(240)
        if not observed.empty:
            chart = px.line(
                observed,
                x=observed.index,
                y=selected_pollutants,
                labels={"value": "Concentration (ug/m3)", "variable": "Pollutant"},
                title="Pollutant history",
            )
            chart.update_layout(legend_title_text="Pollutant")
            st.plotly_chart(chart, use_container_width=True)
            daily = observed.resample("D").mean().round(2)
            st.dataframe(daily.tail(14), use_container_width=True)
            st.download_button(
                "Download observed data as CSV",
                history.to_csv(index=False).encode("utf-8"),
                file_name=f"{resolved_city.lower().replace(' ', '_')}_air_quality.csv",
                mime="text/csv",
            )
        else:
            st.info("More observations are needed before historical charts can be generated.")

    with prediction_tab:
        model, scaler = train_lstm_model(history)
        prediction_mode = st.radio(
            "Prediction method",
            ["Live provider forecast", "Rolling-average baseline", "LSTM when available"],
            index=0,
            horizontal=True,
        )
        if prediction_mode == "Live provider forecast":
            future = provider_forecast_prediction(forecast, forecast_steps)
            if future.empty:
                future = baseline_prediction(history, forecast_steps)
                method_label = "rolling-average fallback"
            else:
                method_label = "live provider"
        elif prediction_mode == "LSTM when available" and model is not None:
            future = predict_future(model, scaler, history, forecast_steps)
            method_label = "LSTM"
        else:
            future = baseline_prediction(history, forecast_steps)
            method_label = "rolling-average baseline"
        observed = history.set_index("datetime")[selected_pollutants].tail(96)
        combined = pd.concat([observed, future[selected_pollutants]])
        prediction_chart = px.line(
            combined,
            x=combined.index,
            y=selected_pollutants,
            labels={"value": "Concentration (ug/m3)", "variable": "Pollutant"},
            title=f"Observed and {method_label} forecast",
        )
        st.plotly_chart(prediction_chart, use_container_width=True)
        if prediction_mode == "Live provider forecast":
            st.caption("This forecast comes from OpenWeather's live three-hour air-pollution forecast.")
        elif model is None:
            st.caption("LSTM is unavailable until TensorFlow is installed and at least 25 observations are stored.")
        else:
            st.caption("LSTM is trained on stored three-hour observations, not provider forecast values.")
        mae = rolling_baseline_mae(history)
        eval_col, next_col = st.columns(2)
        eval_col.metric("Baseline MAE", f"{mae:.2f} ug/m3" if mae is not None else "Collecting data")
        next_col.metric("Predicted PM2.5 peak", f"{future['pm2_5'].max():.1f} ug/m3")

    with assistant_tab:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if use_ai and st.session_state.get("ai_status"):
            st.caption(st.session_state.ai_status)
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(chat["user"])
            with st.chat_message("assistant"):
                st.markdown(chat["assistant"])
        user_message = st.chat_input("Ask about safety, exercise, masks, or pollution")
        if user_message:
            response = chatbot_response(user_message, health_condition, current_aqi, resolved_city)
            if use_ai:
                response = ai_response(user_message, health_condition, current_aqi, resolved_city) or response
                if st.session_state.get("ai_status") != "Gemini response active.":
                    st.warning(st.session_state.get("ai_status", "AI response unavailable; using local fallback."))
            st.session_state.chat_history.append({"user": user_message, "assistant": response})
            st.rerun()


if not API_KEY:
    st.error("Configure OPENWEATHER_API_KEY before starting the application.")
    st.stop()

init_database()
st.title("Air Quality Intelligence")
st.caption("Real-time monitoring, personalized guidance, historical analysis, and transparent forecasting")

with st.sidebar:
    st.header("Controls")
    city = st.text_input("City", placeholder="e.g. London")
    health_condition = st.selectbox(
        "Health profile",
        ["Healthy", "Asthma", "Heart Disease", "Children", "Elderly"],
    )
    history_days = st.slider("History window (days)", 1, 30, 14)
    forecast_steps = st.slider("Forecast horizon (3-hour steps)", 4, 24, 8)
    selected_pollutants = st.multiselect(
        "Pollutants to display",
        POLLUTANT_COLUMNS,
        default=POLLUTANT_COLUMNS[:2],
        format_func=lambda value: POLLUTANT_LABELS[value],
    )
    alert_threshold = st.slider("Alert AQI threshold", 50, 300, 100, step=10)
    use_ai = st.checkbox(
        "Use AI for conversational explanations",
        value=bool(GEMINI_API_KEY or GROK_API_KEY),
        disabled=not bool(GEMINI_API_KEY or GROK_API_KEY),
        help="AQI calculations and safety thresholds always remain local and deterministic.",
    )

if city.strip() and selected_pollutants:
    if hasattr(st, "fragment"):
        render_dashboard = st.fragment(run_every="3h")(render_dashboard)
    render_dashboard(
        city.strip(),
        health_condition,
        history_days,
        forecast_steps,
        selected_pollutants,
        alert_threshold,
        use_ai,
    )
elif not city.strip():
    st.info("Enter a city in the sidebar to open the intelligence dashboard.")
else:
    st.warning("Select at least one pollutant in the sidebar.")
>>>>>>> 36189c1 (Build air quality intelligence dashboard)
