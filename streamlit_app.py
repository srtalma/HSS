import streamlit as st
import pandas as pd
import numpy as np

# Set page title and background image
page_title = "HarmonySplashes (TCS Sustainathon2024)"
st.set_page_config(page_title=page_title)

# Add CSS for background image
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1498928679065-7f48a6bc86d6?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Add title
st.title(page_title)

# Sample data generation using numpy
st.subheader("Sample Data")

# Generate some random data for demonstration purposes
np.random.seed(42)
external_temp = np.random.uniform(10, 35, 100)
room_temp = np.random.uniform(15, 30, 100)
room_humidity = np.random.uniform(30, 70, 100)
flow_rate = np.random.uniform(5, 20, 100)
cold_water_temp = np.random.uniform(5, 25, 100)

# Define desired temperature as a linear combination of the features (for simplicity)
desired_temp = 0.5 * external_temp + 0.3 * room_temp - 0.2 * room_humidity + 0.1 * flow_rate + 0.15 * cold_water_temp + np.random.normal(0, 1, 100)

# Create a DataFrame to display the data
df = pd.DataFrame({
    'ExternalTemp': external_temp,
    'RoomTemp': room_temp,
    'RoomHumidity': room_humidity,
    'FlowRate': flow_rate,
    'ColdWaterTemp': cold_water_temp,
    'DesiredTemp': desired_temp
})

st.dataframe(df.head())

# Linear regression using numpy (no sklearn)
X = np.c_[external_temp, room_temp, room_humidity, flow_rate, cold_water_temp]
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias (intercept) term
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(desired_temp)  # Normal equation

# Prediction function
def predict(features, theta):
    features_b = np.c_[np.ones((features.shape[0], 1)), features]  # Add bias term
    return features_b.dot(theta)

# Predict based on new input data
st.subheader("Predict Desired Temperature based on new input data")

new_data = np.array([[st.number_input('External Temperature', value=15),
                      st.number_input('Room Temperature', value=22),
                      st.number_input('Room Humidity', value=48),
                      st.number_input('Flow Rate', value=9),
                      st.number_input('Cold Water Temperature', value=16)]])

predicted_temp = predict(new_data, theta_best)[0]

st.write(f"**Predicted Desired Temperature: {predicted_temp:.2f}**")

# Model evaluation (for display purposes)
y_pred = predict(X, theta_best)

mse = np.mean((y_pred - desired_temp) ** 2)
r2 = 1 - (np.sum((desired_temp - y_pred) ** 2) / np.sum((desired_temp - np.mean(desired_temp)) ** 2))

st.subheader("Model Evaluation")
st.write(f"Mean Squared Error: {mse}")
st.write(f"R2 Score: {r2}")
