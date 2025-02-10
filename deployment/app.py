import streamlit as st
import pickle
import numpy as np

# Load the model
with open('best_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# List of expected features
expected_features = [
    'International_plan', 'Number_vmail_messages', 'Total_day_minutes',
    'Total_eve_minutes', 'Total_night_minutes', 'Total_intl_minutes',
    'Total_intl_calls', 'Customer_service_calls', 'State_AZ', 'State_CA',
    'State_MD', 'State_MT', 'State_NJ', 'State_SC', 'State_TX'
]

# Streamlit app
st.title("Churn Prediction")

# Custom CSS for the prediction message
st.markdown("""
    <style>
    .prediction {
        font-size: 2em;
        color: #ff4b4b;
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for input fields
st.sidebar.header("Input Features")
input_data = []
for feature in expected_features:
    value = st.sidebar.text_input(feature, '0')
    input_data.append(float(value))

# Predict button
if st.sidebar.button("Predict"):
    final_features = np.array(input_data).reshape(1, -1)
    prediction = model.predict(final_features)
    output = prediction[0]
    
    if output == 1:
        st.markdown('<div class="prediction">Churn Prediction: 1 (The customer is likely to churn)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction">Churn Prediction: 0 (The customer is likely to stay)</div>', unsafe_allow_html=True)