import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
df = pd.read_csv("selected_features_correlation_matrix.csv")

# Step 1: Feature Engineering

# Create a new feature 'car_age' from 'modelYear'
df['car_age'] = 2024 - df['modelYear']

# Create a new feature 'km_per_year' by dividing 'km' by 'car_age'
df['km_per_year'] = df['km'] / df['car_age']

# Interaction feature: Create a feature by multiplying 'Engine Displacement' with 'Mileage'
df['engine_displacement_mileage'] = df['Engine Displacement'] * df['Mileage']

# Log transformation: Apply log transformation to 'km' to reduce skewness
df['log_km'] = np.log1p(df['km'])

# Drop the original 'km' column after log transformation
df = df.drop(columns=['km'])

# Define features and target variable
features = ['car_age', 'km_per_year', 'engine_displacement_mileage', 'log_km',
            'Engine Displacement', 'Mileage', 'modelYear', 'centralVariantId', 'variantName', 
            'ft', 'bt', 'transmission', 'ownerNo', 'oem', 'model']  # Example features
target = 'price'

# Step 2: Imputation and Scaling

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df[features])  # Features

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Train-Test Split

y = df[target]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training with Hyperparameter Tuning

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2', verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Step 5: Model Evaluation

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"R2 Score: {r2}")
print(f"RMSE: {rmse}")

# Save the model
output_path = "C:/Users/Elakkiya/Downloads/gradient_boosting_model_1.joblib"
joblib.dump(best_model, output_path)

print(f"Model saved to {output_path}")
