import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Function to simulate dataset
def generate_data():
    np.random.seed(42)
    data = {
        'UserID': np.arange(1, 101),
        'ExternalTemp': np.random.uniform(0, 40, 100),
        'RoomTemp': np.random.uniform(18, 28, 100),
        'RoomHumidity': np.random.uniform(30, 70, 100),
        'FlowRate': np.random.uniform(5, 20, 100),
        'ColdWaterTemp': np.random.uniform(5, 20, 100),
        'Activity': np.random.choice(['Dishwashing', 'Hand Washing', 'Laundry', 'Shower'], 100),
        'TimeOfDay': np.random.choice(['Morning', 'Afternoon', 'Evening'], 100),
        'Season': np.random.choice(['Autumn', 'Spring', 'Summer', 'Winter'], 100),
        'DesiredTemp': np.random.uniform(20, 30, 100)
    }
    return pd.DataFrame(data)

# Load or create model
def load_or_train_model(df_encoded, y):
    try:
        model = joblib.load('model.pkl')
    except FileNotFoundError:
        # Train the model
        X = df_encoded.drop(['UserID', 'DesiredTemp'], axis=1)
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        joblib.dump(model, 'model.pkl')
    return model

# Function to make a prediction
def predict_temperature(model, input_df):
    return model.predict(input_df)[0]

# Main Streamlit app
st.title("Temperature Prediction App")

# Generate the dataset
df = generate_data()
st.write("Generated Data")
st.dataframe(df.head(10))

# Encode categorical features
encoder = pd.get_dummies(df[['Activity', 'TimeOfDay', 'Season']])
df_encoded = pd.concat([df.drop(['Activity', 'TimeOfDay', 'Season'], axis=1), encoder], axis=1)

# Define target variable
y = df_encoded['DesiredTemp']

# Load or train the model
model = load_or_train_model(df_encoded, y)

# Display form for user input for new prediction
st.subheader("Make a new prediction")
external_temp = st.number_input('External Temperature', min_value=-50, max_value=50, value=15)
room_temp = st.number_input('Room Temperature', min_value=-10, max_value=50, value=22)
room_humidity = st.number_input('Room Humidity', min_value=0, max_value=100, value=48)
flow_rate = st.number_input('Flow Rate', min_value=0, max_value=50, value=9)
cold_water_temp = st.number_input('Cold Water Temperature', min_value=0, max_value=50, value=16)

# Activity, Time of Day, and Season selections
activity = st.selectbox('Activity', ['Dishwashing', 'Hand Washing', 'Laundry', 'Shower'])
time_of_day = st.selectbox('Time of Day', ['Morning', 'Afternoon', 'Evening'])
season = st.selectbox('Season', ['Autumn', 'Spring', 'Summer', 'Winter'])

# One-hot encoding for user input
input_data = {
    'ExternalTemp': [external_temp], 
    'RoomTemp': [room_temp], 
    'RoomHumidity': [room_humidity], 
    'FlowRate': [flow_rate],
    'ColdWaterTemp': [cold_water_temp],
    f'Activity_{activity}': [1],
    f'TimeOfDay_{time_of_day}': [1],
    f'Season_{season}': [1]
}

# Handle missing one-hot encoded columns
activities = ['Dishwashing', 'Hand Washing', 'Laundry', 'Shower']
times_of_day = ['Morning', 'Afternoon', 'Evening']
seasons = ['Autumn', 'Spring', 'Summer', 'Winter']

for act in activities:
    if f'Activity_{act}' not in input_data:
        input_data[f'Activity_{act}'] = [0]
for tod in times_of_day:
    if f'TimeOfDay_{tod}' not in input_data:
        input_data[f'TimeOfDay_{tod}'] = [0]
for sea in seasons:
    if f'Season_{sea}' not in input_data:
        input_data[f'Season_{sea}'] = [0]

input_df = pd.DataFrame(input_data)

# Prediction
if st.button("Predict"):
    predicted_temp = predict_temperature(model, input_df)
    st.success(f'Predicted Desired Temperature: {predicted_temp:.2f}°C')
