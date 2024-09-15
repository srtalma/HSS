import streamlit as st
import pandas as pd
import numpy as np

# Generate sample data (instead of uploading)
np.random.seed(42)
n_samples = 100
external_temp = np.random.uniform(10, 35, n_samples)
room_temp = np.random.uniform(15, 30, n_samples)
room_humidity = np.random.uniform(40, 60, n_samples)
flow_rate = np.random.uniform(5, 15, n_samples)
cold_water_temp = np.random.uniform(10, 20, n_samples)
desired_temp = 0.5 * external_temp + 0.3 * room_temp - 0.1 * room_humidity + flow_rate * 0.2 + cold_water_temp * 0.15 + np.random.normal(0, 1, n_samples)

# Create a DataFrame
df = pd.DataFrame({
    'ExternalTemp': external_temp,
    'RoomTemp': room_temp,
    'RoomHumidity': room_humidity,
    'FlowRate': flow_rate,
    'ColdWaterTemp': cold_water_temp,
    'DesiredTemp': desired_temp
})

# Display the first few rows of the generated data
st.write("Sample Data")
st.dataframe(df.head())

# Prepare the features and the target variable
X = df[['ExternalTemp', 'RoomTemp', 'RoomHumidity', 'FlowRate', 'ColdWaterTemp']].values
y = df['DesiredTemp'].values

# Train a simple linear regression model using NumPy
X_b = np.c_[np.ones((n_samples, 1)), X]  # Add bias (intercept) term
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Predict function
def predict(features, theta):
    features_b = np.c_[np.ones((features.shape[0], 1)), features]  # Add bias term
    return features_b.dot(theta)

# Make predictions on the generated data
y_pred = predict(X, theta_best)

# Display Mean Squared Error
mse = np.mean((y_pred - y) ** 2)
st.write(f'Mean Squared Error: {mse}')

# Example for prediction
st.write("Predict Desired Temperature based on new input data")
external_temp_input = st.number_input("External Temperature", value=15)
room_temp_input = st.number_input("Room Temperature", value=22)
room_humidity_input = st.number_input("Room Humidity", value=50)
flow_rate_input = st.number_input("Flow Rate", value=10)
cold_water_temp_input = st.number_input("Cold Water Temperature", value=15)

# Prepare input for prediction
new_data = np.array([[external_temp_input, room_temp_input, room_humidity_input, flow_rate_input, cold_water_temp_input]])
predicted_temp = predict(new_data, theta_best)

st.write(f'Predicted Desired Temperature: {predicted_temp[0]}')
