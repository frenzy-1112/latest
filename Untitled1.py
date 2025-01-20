#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st
import numpy as np
import tensorflow as tf
import requests

# Load the trained LSTM model
model = tf.keras.models.load_model('improved_roulette_lstm_model_with_column.h5')

# Compile the model to avoid the warning about uncompiled metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to fetch the last 10 spins from the Tracksino API
def fetch_last_10_spins():
    url = "https://api.tracksino.com/lightningroulette_history"
    params = {
        'sort_by': '',
        'sort_desc': 'false',
        'page_num': '1',
        'per_page': '10',
        'period': '1hour',  # Ensure this is the correct format expected by the API
        'table_id': '4'
    }
    headers = {
        'Authorization': 'Bearer 35423482-f852-453c-97a4-4f5763f4796f',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Ch-Ua': '"Not A(Brand";v="8", "Chromium";v="132", "Brave";v="132"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Gpc': '1',
        'Accept-Language': 'en-US,en;q=0.5',
        'Origin': 'https://tracksino.com',
        'Sec-Fetch-Site': 'same-site',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
        'Referer': 'https://tracksino.com/',
        'Accept-Encoding': 'gzip, deflate, br'
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        try:
            # Extract the 'result' field from the API response
            data = [entry['result'] for entry in response.json().get('data', [])]
            
            # Replace any occurrence of 37 with 0
            data = [number if number != 37 else 0 for number in data]
            
            return data
        except KeyError as e:
            st.error(f"Error processing data: {e}")
            return []
    else:
        st.error(f"Failed to fetch data. Status code: {response.status_code}")
        return []

# Streamlit UI
st.title("Roulette Prediction with LSTM")
st.write("Enter the last 10 roulette numbers or let the app fetch them from the API.")

# Fetch the last 10 spins from the Tracksino API
if st.button("Fetch Last 10 Spins"):
    with st.spinner('Fetching data from the API...'):
        last_10_spins = fetch_last_10_spins()

    if last_10_spins:
        # Reverse the order of the fetched spins
        last_10_spins.reverse()

        st.write("Last 10 Spins (Reversed Order):", last_10_spins)

        # Predict and display the results
        predicted_dozens = predict_next_two_dozens(last_10_spins)
        st.write(f"Next Possible Winning Dozens: {predicted_dozens}")
    else:
        st.error("Could not fetch the last 10 spins.")


# In[ ]:




