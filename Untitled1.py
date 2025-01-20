#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import tensorflow as tf
import requests

# Load the trained LSTM model
model = tf.keras.models.load_model('improved_roulette_lstm_model_with_column.h5')

# Compile the model to avoid the warning about uncompiled metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to map number to color, parity, high/low, and dozen
def get_color(number):
    red_numbers = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
    if number == 0:
        return 2  # Green
    elif number in red_numbers:
        return 1  # Red
    else:
        return 0  # Black

def get_parity(number):
    return 1 if number % 2 == 0 else 0  # Even = 1, Odd = 0

def get_high_low(number):
    return 1 if 1 <= number <= 18 else 0  # Low = 1, High = 0

def get_dozen(number):
    if 1 <= number <= 12:
        return 0  # Dozen 1
    elif 13 <= number <= 24:
        return 1  # Dozen 2
    elif 25 <= number <= 36:
        return 2  # Dozen 3
    else:
        return 3  # Green (0)

def get_section(number):
    voisins = {22, 18, 29, 7, 28, 12, 35, 3, 26, 0, 32, 15, 19, 4, 21, 2, 25}
    tiers = {27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33}
    orphelins = {17, 34, 6, 1, 20, 14, 31, 9}
    
    if number in voisins:
        return 0  # Voisins
    elif number in tiers:
        return 1  # Tiers
    elif number in orphelins:
        return 2  # Orphelins
    else:
        return 3  # Green (0)

def get_column(number):
    if number == 0:
        return 0  # For number 0, set column as 0
    else:
        return (number - 1) % 3 + 1

def get_neighbors(number):
    wheel_neighbors = [
        0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
    ]
    i = wheel_neighbors.index(number)
    return wheel_neighbors[(i - 1) % len(wheel_neighbors)], wheel_neighbors[(i + 1) % len(wheel_neighbors)]


# Predict the next two winning dozens based on the last 10 spins
def predict_next_two_dozens(last_10_spins):
    # Prepare input features by mapping numbers to their corresponding features
    input_features = []
    for number in last_10_spins:
        neighbors_1, neighbors_2 = get_neighbors(number)
        input_features.append([
            number,  # Roulette number
            get_color(number),  # Color (Red, Black, Green)
            get_parity(number),  # Parity (Even, Odd)
            get_high_low(number),  # High/Low (1-18, 19-36)
            neighbors_1,  # Neighbor 1
            neighbors_2,  # Neighbor 2
            get_section(number),  # Section (Voisins, Tiers, Orphelins)
            get_column(number)  # Column (1, 2, or 3)
        ])

    # Reshape input to match LSTM input shape (1, sequence_length, num_features)
    input_features = np.array(input_features).reshape(1, len(last_10_spins), 8)
    
    # Get probabilities for each dozen
    probabilities = model.predict(input_features)[0]
    
    # Get the indices of the top 2 most probable dozens
    top_2_indices = probabilities.argsort()[-2:][::-1]
    
    # Map indices to dozen names
    dozen_mapping = {0: "Dozen 1 (1-12)", 1: "Dozen 2 (13-24)", 2: "Dozen 3 (25-36)", 3: "Green (0)"}
    return [dozen_mapping[idx] for idx in top_2_indices]

# Function to fetch the last 10 spins from the new API
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
            
            # Replace any occurrence of 37 with 0 (roulette 37 becomes 0)
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

# Fetch the last 10 spins from the new API
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




