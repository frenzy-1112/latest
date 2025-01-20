#!/usr/bin/env python
# coding: utf-8

# In[9]:


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
    base_url = "https://api.casinoscores.com/svc-evolution-game-events/api/lightningroulette"
    params = {
        "size": 10,  # Fetch only the last 10 spins
        "sort": "data.settledAt,desc",  # Sort by descending settlement time
        "duration": 72,  # Optional: Filter based on the duration (72 hours)
        "isLightningNumberMatched": "false"  # Optional: Filter out lightning numbers
    }
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        numbers = [
            event["data"]["result"]["outcome"]["number"]
            for event in data
            if "data" in event and "result" in event["data"] and "outcome" in event["data"]["result"]
        ]
        return numbers
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




