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

# Function to generate additional random data
def generate_random_data(num_samples=10):
    activities = ['Shower', 'Hand Washing', 'Dishwashing', 'Laundry']
    times_of_day = ['Morning', 'Afternoon', 'Evening']
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']

    data = {
        'UserID': np.arange(1, num_samples + 1),
        'Activity': np.random.choice(activities, num_samples),
        'TimeOfDay': np.random.choice(times_of_day, num_samples),
        'Season': np.random.choice(seasons, num_samples),
        'ExternalTemp': np.random.randint(-5, 40, num_samples),
        'RoomTemp': np.random.randint(15, 30, num_samples),
        'RoomHumidity': np.random.randint(40, 60, num_samples),
        'FlowRate': np.random.randint(5, 15, num_samples),
        'ColdWaterTemp': np.random.randint(5, 25, num_samples),
        'DesiredTemp': np.random.randint(28, 45, num_samples)
    }

    return pd.DataFrame(data)

# Generate and display random sample data
df = generate_random_data(10)
st.subheader("Generated Sample Data")
st.dataframe(df)

# Predict based on new input data using numpy (no ML model)
st.subheader("Predict Desired Temperature based on new input data")

ext_temp = st.number_input('External Temperature', value=15)
room_temp = st.number_input('Room Temperature', value=22)
room_humidity = st.number_input('Room Humidity', value=48)
flow_rate = st.number_input('Flow Rate', value=9)
cold_water_temp = st.number_input('Cold Water Temperature', value=16)

# Basic numpy-based prediction (for illustration)
predicted_temp = (0.5 * ext_temp + 0.3 * room_temp + 0.1 * room_humidity + 
                  0.05 * flow_rate + 0.05 * cold_water_temp) / 2

st.write(f"**Predicted Desired Temperature: {predicted_temp:.2f}**")
