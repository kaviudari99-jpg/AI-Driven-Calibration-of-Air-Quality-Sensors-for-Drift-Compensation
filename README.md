# AI-Driven-Calibration-of-Air-Quality-Sensors-for-Drift-Compensation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. DATA PREPROCESSING
print("Loading data...")
# Load the dataset (Ensure this file is in your project folder)
df = pd.read_csv(r'd:/YEAR 01/SEM 02/DEEP LEARNING/PROJECTS/Measurement_info.csv')

# Focus on PM2.5 (Tiny dust particles)
pm25 = df[df['Item code'] == 9]

print("Processing Truth and Drift pairs...")
# Group A: Calculate 'Global Truth' (Average of all healthy sensors per hour)
truth_all = pm25[pm25['Instrument status'] == 0]
city_truth = truth_all.groupby('Measurement date')['Average value'].mean().reset_index()
city_truth.columns = ['Measurement date', 'Average value_Truth']

# Group B: Collect all sensors flagged as needing calibration
drift_data = pm25[pm25['Instrument status'] == 1][['Measurement date', 'Average value']]
drift_data.columns = ['Measurement date', 'Average value_Drift']

# Merge by Time: This pairs every 'bad' reading with the 'city average' at that moment
pairs = pd.merge(city_truth, drift_data, on='Measurement date')

if pairs.empty:
    print("Error: No overlapping timestamps found between Status 0 and Status 1.")
    exit()

print(f"Success! Found {len(pairs)} calibration pairs.")

# 2. NORMALIZATION
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# Reshape data for scaling (Needs to be 2D)
X_raw = pairs[['Average value_Drift']].values
y_raw = pairs[['Average value_Truth']].values

X_scaled = scaler_x.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 3. DEEP LEARNING MODEL DESIGN
model = tf.keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1) # Final prediction (the corrected value)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 4. TRAINING
print("Training the Deep Learning Calibration model...")
# Increased epochs to 100 for better convergence since we have more data now
history = model.fit(X_train, y_train, validation_split=0.2, epochs=500, batch_size=64, verbose=1)

# 5. GENERATING RESULTS
# Use the model to predict the corrected values
predictions_scaled = model.predict(X_scaled)
# Convert back from 0-1 scale to real-world units
pairs['Corrected Value'] = scaler_y.inverse_transform(predictions_scaled).flatten()

# Calculate Errors for comparison
pairs['Raw Error'] = (pairs['Average value_Drift'] - pairs['Average value_Truth']).round(2)
pairs['Residual Error'] = (pairs['Corrected Value'] - pairs['Average value_Truth']).round(2)

# 6. SUMMARY & VISUALIZATION
print("\n--- CALIBRATION SUMMARY TABLE (First 10 Pairs) ---")
print(pairs[['Measurement date', 'Average value_Truth', 'Average value_Drift', 'Corrected Value', 'Residual Error']].head(10).to_string(index=False))

# Performance Check
print(f"\nAverage Error BEFORE Calibration: {pairs['Raw Error'].abs().mean():.4f}")
print(f"Average Error AFTER Calibration: {pairs['Residual Error'].abs().mean():.4f}")

# Plotting the Learning Curve
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Deep Learning Sensor Calibration: Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
# Focus on PM2.5 (Item Code 9)
pm25 = df[df['Item code'] == 9]

# Separate Truth (Status 0) and Drift (Status 1)
truth = pm25[pm25['Instrument status'] == 0][['Measurement date', 'Average value', 'Station code']]
drift = pm25[pm25['Instrument status'] == 1][['Measurement date', 'Average value', 'Station code']]

# Merge to create pairs (Matching by Time and Station for better accuracy)
pairs = pd.merge(truth, drift, on=['Measurement date', 'Station code'], suffixes=('_Truth', '_Drift'))

if pairs.empty:
    print("No matching pairs found. Check your dataset statuses.")
    exit()

# 2. NORMALIZATION
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# Reshape for the scaler
X_raw = pairs[['Average value_Drift']].values
y_raw = pairs[['Average value_Truth']].values

X_scaled = scaler_x.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 3. DEEP LEARNING MODEL DESIGN
model = tf.keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1) # Linear output for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 4. TRAINING
print("Training the Deep Learning Calibration model...")
history = model.fit(X_train, y_train, validation_split=0.2, epochs=500, batch_size=32, verbose=0)

# 5. EVALUATION & INVERSE SCALING
predictions_scaled = model.predict(X_scaled)
# Convert back to original units (e.g., PM2.5 concentration)
pairs['Corrected Value'] = scaler_y.inverse_transform(predictions_scaled).flatten()

# Calculate Errors
pairs['Raw Error'] = (pairs['Average value_Drift'] - pairs['Average value_Truth']).round(2)
pairs['Residual Error'] = (pairs['Corrected Value'] - pairs['Average value_Truth']).round(2)

# 6. RESULTS & VISUALIZATION
print("\n--- CALIBRATION SUMMARY ---")
print(pairs[['Measurement date', 'Average value_Truth', 'Average value_Drift', 'Corrected Value', 'Residual Error']].head(10).to_string(index=False))

print(f"\nFinal Average Error after ML: {pairs['Residual Error'].abs().mean():.4f}")

# Plot Learning Curve
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Learning Progress')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epochs')
plt.legend()
plt.show()
pm25 = df[df['Item code'] == 9]

# 2. Separate (Make sure to keep 'Measurement date' for the merge!)
truth = pm25[pm25['Instrument status'] == 0][['Measurement date', 'Average value']]
drift = pm25[pm25['Instrument status'] == 1][['Measurement date', 'Average value']]

# 3. Merge carefully
# We merge on date so each row has the Truth and the Drift at the same time
pairs = pd.merge(truth, drift, on='Measurement date', suffixes=('_Truth', '_Drift'))

# --- DEBUG STEP ---
# If this prints 0, your merge didn't find any matching timestamps!
print(f"Number of pairs found: {len(pairs)}") 
print(f"Available columns: {pairs.columns.tolist()}")
# ------------------

# 4. Use the correct names found in the debug step
# Make sure the strings match exactly what printed above
X = pairs[['Average value_Drift']] 
y = pairs['Average value_Truth']

# 5. ML Steps 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print("Calibration Model Trained Successfully!")

# 1. Get the model's predictions (The 'Fixed' values)
y_pred = model.predict(X_test)

# 2. Calculate the "New" error after calibration
final_error = mean_absolute_error(y_test, y_pred)
accuracy = r2_score(y_test, y_pred)

print(f"Average error after calibration: {final_error:.2f} units")
print(f"Calibration Accuracy (R2): {accuracy:.2%}")

# 1. Use the trained model to predict 'Corrected' values for the whole dataset
# (Assuming your model is named 'model' and your merged data is 'pairs')
pairs['Corrected Value'] = model.predict(pairs[['Average value_Drift']]).round(2)

# 2. Calculate the Errors
# Raw Error = Drifted Sensor - Truth
pairs['Raw Error'] = (pairs['Average value_Drift'] - pairs['Average value_Truth']).round(2)

# Residual Error = ML Output - Truth
pairs['Residual Error'] = (pairs['Corrected Value'] - pairs['Average value_Truth']).round(2)
summary_table = pairs[[
    'Measurement date', 
    'Average value_Truth', 
    'Average value_Drift', 
    'Raw Error', 
    'Corrected Value', 
    'Residual Error'
]]

# 4. Print the table to confirm
print("--- CALIBRATION SUMMARY TABLE ---")
print(summary_table.head(10).to_string(index=False))

# 5. Final Statistics (Useful for your report)
print("\n--- PERFORMANCE METRICS ---")
print(f"Average Raw Error: {pairs['Raw Error'].abs().mean():.2f}")
print(f"Average Error After ML: {pairs['Residual Error'].abs().mean():.2f}")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Define the Deep Learning Model
model = tf.keras.Sequential([
    layers.Input(shape=(1,)),               # Input: The drifting sensor value
    layers.Dense(64, activation='relu'),    # Hidden Layer 1: 64 neurons
    layers.Dense(32, activation='relu'),    # Hidden Layer 2: 32 neurons
    layers.Dense(1)                         # Output: The corrected value
])

# 2. Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. Train the model
# epochs=50 means the model will look at the data 50 times
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

# 4. Use it to get the 'Corrected Value'
pairs['Corrected Value'] = model.predict(pairs[['Average value_Drift']]).flatten()

import matplotlib.pyplot as plt

# Plotting the Training vs Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Deep Learning Calibration: Model Learning Curve')
plt.xlabel('Epochs (Training Rounds)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()
