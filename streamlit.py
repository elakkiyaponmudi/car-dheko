import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import base64

# Load the pre-trained model
model = joblib.load('C:/Users/Elakkiya/Downloads/gradient_boosting_model_1.joblib')

# Load the dataset to get unique values for dropdowns
data = pd.read_csv("C:/Users/Elakkiya/Downloads/dropped_dataset.csv")

# Ensure the relevant columns are numeric
data['Engine Displacement'] = pd.to_numeric(data['Engine Displacement'], errors='coerce')
data['Mileage'] = pd.to_numeric(data['Mileage'], errors='coerce')

# Drop rows with NaN values in these columns if needed, or fill NaN with a default value
data.dropna(subset=['Engine Displacement', 'Mileage'], inplace=True)

# Initialize LabelEncoder and MinMaxScaler
label_encoders = {}
scalers = {}

# Create LabelEncoders for categorical features
categorical_features = ['ft', 'bt', 'transmission', 'ownerNo', 'oem', 'model', 'variantName', 'city']
for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data[feature].astype(str))
    label_encoders[feature] = le

# Create MinMaxScaler for numerical features
numerical_features = ['modelYear','Mileage', 'car_age', 'km_per_year', 'engine_displacement_mileage', 'log_km']
data['car_age'] = 2024 - data['modelYear']
data['km_per_year'] = data['km'] / data['car_age']
data['engine_displacement_mileage'] = data['Engine Displacement'] * data['Mileage']
data['log_km'] = np.log1p(data['km'])
data = data.drop(columns=['km'])
scalers['numerical'] = MinMaxScaler()
scalers['numerical'].fit(data[numerical_features])

# Function to set background image
def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
        }}
        .stSidebar {{
            background-color: black;
            color: red;
        }}
        .stButton button {{
            background-color: red;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Main function
def main():
    # Set the background image (provide the path to your image)
    set_background_image(r"C:\Users\Elakkiya\Downloads\cd_mall_opt.medium.jpg")  # Replace with your image file path

    # Display the title with white color
    st.markdown("<h1 style='color: white;'>Used Car Price Prediction</h1>", unsafe_allow_html=True)

    # Sidebar input fields for specific features
    with st.sidebar:
        ft = st.selectbox("Fuel Type", options=sorted(data['ft'].dropna().unique()))
        bt = st.selectbox("Body Type", options=sorted(data['bt'].dropna().unique()))
        transmission = st.selectbox("Transmission", options=sorted(data['transmission'].dropna().unique()))
        ownerNo = st.selectbox("Owner Number", options=sorted(data['ownerNo'].dropna().unique()))
        oem = st.selectbox("OEM", options=sorted(data['oem'].dropna().unique()))
        model_selected = st.selectbox("Model", options=sorted(data['model'].dropna().unique()))
        modelYear = st.selectbox("Model Year", options=sorted(data['modelYear'].dropna().unique()))
        variantName = st.selectbox("Variant Name", options=sorted(data['variantName'].dropna().unique()))
        
        
        # Slider options for Mileage and Engine Displacement
        Mileage = st.slider("Mileage (in kmpl)", min_value=float(data['Mileage'].min()), max_value=float(data['Mileage'].max()), value=float(data['Mileage'].mean()))
        Engine_Displacement = st.slider("Engine Displacement (in cc)", min_value=float(data['Engine Displacement'].min()), max_value=float(data['Engine Displacement'].max()), value=float(data['Engine Displacement'].mean()))
        
        city = st.selectbox("City", options=sorted(data['city'].dropna().unique()))
        car_age = 2024 - int(modelYear)  # Calculate car age

        # Calculate min and max values for kilometers driven per year
        min_km_per_year = int(data['km_per_year'].min())
        max_km_per_year = int(data['km_per_year'].max())

        # Ensure min_km_per_year and max_km_per_year are valid for the slider
        if min_km_per_year == max_km_per_year:
            min_km_per_year = 0  # Set a sensible default if min and max are the same
            max_km_per_year = 100000  # Arbitrary large number for the upper limit

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
        'variantName': [variantName],
        'Engine Displacement': [Engine_Displacement],
        'Mileage': [Mileage],
        'city': [city],
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
            st.markdown(f"<h2 style='color: white;'>Estimated Price: {prediction[0]:,.2f}</h2>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Run the main function
if __name__ == "__main__":
    main()

















 
