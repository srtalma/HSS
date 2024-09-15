import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

# Sample data (for illustration purposes)
st.subheader("Sample Data")

# Create and display some dummy data
data = {
    'ExternalTemp': [10, 15, 20],
    'RoomTemp': [22, 23, 21],
    'RoomHumidity': [50, 45, 55],
    'FlowRate': [8, 9, 7],
    'ColdWaterTemp': [12, 16, 18],
    'DesiredTemp': [16.3, 17.5, 18.2]
}

df = pd.DataFrame(data)
st.dataframe(df)

# Example of model training (RandomForestRegressor)
X = df.drop(columns=['DesiredTemp'])
y = df['DesiredTemp']

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Predict on new data
st.subheader("Predict Desired Temperature based on new input data")

new_data = {
    'ExternalTemp': st.number_input('External Temperature', value=15),
    'RoomTemp': st.number_input('Room Temperature', value=22),
    'RoomHumidity': st.number_input('Room Humidity', value=48),
    'FlowRate': st.number_input('Flow Rate', value=9),
    'ColdWaterTemp': st.number_input('Cold Water Temperature', value=16)
}

input_df = pd.DataFrame([new_data])
predicted_temp = model.predict(input_df)[0]

st.write(f"**Predicted Desired Temperature: {predicted_temp:.2f}**")

# Model evaluation (for display purposes)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.subheader("Model Evaluation")
st.write(f"Mean Squared Error: {mse}")
st.write(f"R2 Score: {r2}")
