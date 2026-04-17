# 1. Import libraries
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 2. Title
st.title("🚦 Traffic Speed Prediction App")

# 3. Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("delhi_traffic_features.csv")
    return df

df = load_data()

# 4. Preprocessing
df = df.drop(columns=["Trip_ID"])

label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 5. Features & target
X = df.drop("average_speed_kmph", axis=1)
y = df["average_speed_kmph"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------- UI ---------------- #

st.sidebar.header("Enter Trip Details")

# Inputs (adjust based on your dataset values)
distance = st.sidebar.number_input("Distance (km)", min_value=0.1, value=5.0)
duration = st.sidebar.number_input("Duration (min)", min_value=1.0, value=15.0)

traffic = st.sidebar.selectbox("Traffic Density", label_encoders["Traffic_Density"].classes_)
weather = st.sidebar.selectbox("Weather Conditions", label_encoders["Weather_Conditions"].classes_)
road = st.sidebar.selectbox("Road Type", label_encoders["Road_Type"].classes_)
time = st.sidebar.selectbox("Time of Day", label_encoders["Time_of_Day"].classes_)

# Encode user input
input_data = pd.DataFrame({
    "distance_km": [distance],
    "duration_min": [duration],
    "Traffic_Density": [label_encoders["Traffic_Density"].transform([traffic])[0]],
    "Weather_Conditions": [label_encoders["Weather_Conditions"].transform([weather])[0]],
    "Road_Type": [label_encoders["Road_Type"].transform([road])[0]],
    "Time_of_Day": [label_encoders["Time_of_Day"].transform([time])[0]],
})

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)

# Output
st.subheader("🚗 Predicted Average Speed")
st.success(f"{prediction[0]:.2f} km/h")
