# -*- coding: utf-8 -*-
"""streamlit_app

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qJnVfyE2N7FbuMxn16TkYwvph9X6fmDE
"""

import streamlit as st
import pandas as pd
import joblib

# Load your pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('model.pkl')
    return model

# Prediction function
def predict_temperature(input_data):
    model = load_model()
    predicted_temp = model.predict(input_data)
    return predicted_temp[0]

# Main function for Streamlit app
def main():
    st.title("Desired Temperature Prediction App")

    # Upload Excel or CSV file section
    uploaded_file = st.file_uploader("Upload your data (Excel or CSV)", type=['xlsx', 'csv'])

    if uploaded_file is not None:
        # Read file
        try:
            if uploaded_file.name.endswith('xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            st.write("Uploaded Data")
            st.dataframe(df.head(10))

            # Process and drop NaNs
            df = df.dropna()

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

            # Handle missing activity, time, and season values in the input
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
                predicted_temp = predict_temperature(input_df)
                st.success(f'Predicted Desired Temperature: {predicted_temp:.2f}°C')

    else:
        st.write("Please upload a CSV or Excel file.")

if __name__ == "__main__":
    main()