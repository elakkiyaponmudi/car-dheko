import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the pre-trained model
model = joblib.load("C:/Users/Elakkiya/Downloads/gradient_boosting_model.joblib")

# Load the dataset to get unique values for dropdowns
data = pd.read_csv("C:/Users/Elakkiya/Downloads/dropped_dataset.csv")

# Initialize LabelEncoder and MinMaxScaler
label_encoders = {}
scalers = {}

# Create LabelEncoders for categorical features
categorical_features = ['ft', 'bt', 'transmission', 'ownerNo', 'oem', 'model', 'centralVariantId', 'variantName']
for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data[feature].astype(str))
    label_encoders[feature] = le

# Create MinMaxScaler for numerical features
numerical_features = ['Engine Displacement', 'Mileage', 'car_age', 'km_per_year', 'engine_displacement_mileage', 'log_km']
scalers['numerical'] = MinMaxScaler()
data['car_age'] = 2024 - data['modelYear']
data['km_per_year'] = data['km'] / data['car_age']
data['engine_displacement_mileage'] = data['Engine Displacement'] * data['Mileage']
data['log_km'] = np.log1p(data['km'])
data = data.drop(columns=['km'])
scalers['numerical'].fit(data[numerical_features])

# Streamlit app interface
st.title("Used Car Price Prediction")

# Sidebar input fields for specific features
with st.sidebar:
    ft = st.selectbox("Fuel Type", options=sorted(data['ft'].dropna().unique()))
    bt = st.selectbox("Body Type", options=sorted(data['bt'].dropna().unique()))
    transmission = st.selectbox("Transmission", options=sorted(data['transmission'].dropna().unique()))
    ownerNo = st.selectbox("Owner Number", options=sorted(data['ownerNo'].dropna().unique()))
    oem = st.selectbox("OEM", options=sorted(data['oem'].dropna().unique()))
    model_selected = st.selectbox("Model", options=sorted(data['model'].dropna().unique()))
    modelYear = st.selectbox("Model Year", options=sorted(data['modelYear'].dropna().unique()))
    centralVariantId = st.selectbox("Central Variant ID", options=sorted(data['centralVariantId'].dropna().unique()))
    variantName = st.selectbox("Variant Name", options=sorted(data['variantName'].dropna().unique()))
    Engine_Displacement = st.selectbox("Engine Displacement", options=sorted(data['Engine Displacement'].dropna().unique()))
    Mileage = st.selectbox("Mileage", options=sorted(data['Mileage'].dropna().unique()))
    car_age = 2024 - int(modelYear)  # Calculate car age

    # Calculate min and max values for kilometers driven per year
    min_km_per_year = int(data['km_per_year'].min())
    max_km_per_year = int(data['km_per_year'].max())

    # Input for kilometers driven per year
    km_per_year = st.slider(
        "Kilometers Driven per Year",
        min_value=min_km_per_year,
        max_value=max_km_per_year,
        value=min_km_per_year
    )

# Calculate additional features
engine_displacement_mileage = Engine_Displacement * Mileage
log_km = np.log1p(km_per_year)

# Prepare input DataFrame with the transformed features
input_data = {
    'ft': [ft],
    'bt': [bt],
    'transmission': [transmission],
    'ownerNo': [ownerNo],
    'oem': [oem],
    'model': [model_selected],
    'modelYear': [modelYear],
    'centralVariantId': [centralVariantId],
    'variantName': [variantName],
    'Engine Displacement': [Engine_Displacement],
    'Mileage': [Mileage],
    'car_age': [car_age],
    'km_per_year': [km_per_year],
    'engine_displacement_mileage': [engine_displacement_mileage],
    'log_km': [log_km]
}
input_df = pd.DataFrame(input_data)

# Perform label encoding for categorical features
for feature in categorical_features:
    input_df[feature] = label_encoders[feature].transform(input_df[feature].astype(str))

# Perform Min-Max scaling for numerical features
input_df[numerical_features] = scalers['numerical'].transform(input_df[numerical_features])

# Main page prediction button and result display
if st.button('Predict Price'):
    try:
        prediction = model.predict(input_df)
        st.subheader(f"Estimated Price: {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
