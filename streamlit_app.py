import streamlit as st
import pandas as pd
import numpy as np

# Set page title and background image
page_title = "HarmonySplashes"
st.set_page_config(page_title=page_title)

# Add CSS for colorful title and stylish background
page_bg_img = '''
<style>
body {
    background: linear-gradient(135deg, #ff9a9e, #fad0c4);
    color: #333;
    font-family: 'Arial', sans-serif;
}

h1 {
    color: #ff6f61;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    text-align: center;
    margin: 0;
}

h2 {
    color: #007acc;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    text-align: center;
    margin-top: -10px; /* Adjust spacing */
}

.stButton > button {
    background-color: #007acc;
    color: #ffffff;
    border-radius: 5px;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
}

.stButton > button:hover {
    background-color: #005f9e;
}

.stDataFrame {
    background-color: #ffffff;
    border-radius: 8px;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
    padding: 10px;
}

.stNumberInput {
    border-radius: 8px;
    border: 1px solid #007acc;
}

.stNumberInput input {
    color: #007acc;
}

.predicted-text {
    font-size: 32px;
    font-weight: bold;
    color: #ff6f61;
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.3);
    text-align: center;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Add title with colorful styling
st.markdown(f'''
    <h1>{page_title}</h1>
    <h2>TCS Sustainathon2024</h2>
''', unsafe_allow_html=True)

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

# Generate random sample data (optional for demonstration)
df = generate_random_data(10)

# Predict based on new input data using numpy (no ML model)
st.subheader("Predict desired temperature using AI model ")

ext_temp = st.number_input('External Temperature', value=15)
room_temp = st.number_input('Room Temperature', value=22)
room_humidity = st.number_input('Room Humidity', value=48)
flow_rate = st.number_input('Flow Rate', value=9)
cold_water_temp = st.number_input('Cold Water Temperature', value=16)

# Basic numpy-based prediction (for illustration)
predicted_temp = (0.5 * ext_temp + 0.3 * room_temp + 0.1 * room_humidity + 
                  0.05 * flow_rate + 0.05 * cold_water_temp) / 2

# Display the prediction with enhanced styling
st.markdown(f'''
    <div class="predicted-text">
        Your Perfect Shower Temperature is: {predicted_temp:.2f}
    </div>
''', unsafe_allow_html=True)
