import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model & preprocessing objects
model = joblib.load("svm_model.pkl")  
scaler = joblib.load("scaler.pkl")  
label_encoder = joblib.load("label_encoder.pkl")  # Load target label encoder
feature_names = joblib.load("feature_names.pkl")  # Load feature names for correct order

# Load dataset for reference
dataset_path = "200125_LoL_champion_data.csv"  
df = pd.read_csv(dataset_path)

st.title("League of Legends Hero/Champion Role Predictor")

# --- INPUT FIELDS (Matches Dataset Features) ---
input_data_dict = {feature: 0 for feature in feature_names}  # Default all features to 0

# Get categorical columns used for one-hot encoding
categorical_cols = ["herotype", "resource", "rangetype"]

# Handle categorical inputs dynamically
for feature in categorical_cols:
    user_input = st.selectbox(f"Select {feature}:", df[feature].unique())
    
    # One-hot encoding: Set the selected category to 1, others remain 0
    for category in df[feature].unique():
        column_name = f"{feature}_{category}"  # Example: "herotype_Assassin"
        if column_name in feature_names:  # Only update if it was used in training
            input_data_dict[column_name] = 1 if user_input == category else 0

# Handle numerical inputs
numeric_features = [f for f in feature_names if f not in input_data_dict]  # Get remaining numeric columns

for feature in numeric_features:
    user_input = st.number_input(f"Enter {feature}:", min_value=float(df[feature].min()), 
                                 max_value=float(df[feature].max()), value=float(df[feature].mean()))
    input_data_dict[feature] = user_input

# Convert to DataFrame and ensure order is correct
input_data = pd.DataFrame([input_data_dict])[feature_names]

# Scale the input
input_scaled = scaler.transform(input_data)

# --- PREDICTION ---
if st.button("Predict Role"):
    prediction_encoded = model.predict(input_scaled)
    prediction = label_encoder.inverse_transform(prediction_encoded)  # Decode predicted role
    st.write(f"### Predicted Role: {prediction[0]}")

# --- VISUALIZATIONS (EXACTLY FROM YOUR NOTEBOOK) ---
st.header("Hero/Champion Data Visualizations")

# **Champion Types Count**
st.subheader("Hero Type Count")
fig, ax = plt.subplots()
df['herotype'].value_counts().plot(kind='bar', color='lightblue', edgecolor='black', ax=ax)
ax.set_xlabel("Hero Type")
ax.set_ylabel("Count")
ax.set_title("Hero Type Distribution")
st.pyplot(fig)

# **Champion Roles Count**
st.subheader("Hero Role Count")
fig, ax = plt.subplots()
df['role'].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black', ax=ax)
ax.set_xlabel("Role")
ax.set_ylabel("Count")
ax.set_title("Hero Role Distribution")
st.pyplot(fig)

# **Pie Chart (Now Exactly Matches Your Notebook Style)**
st.subheader("Hero Difficulty Distribution")

# Mapping numerical difficulty values to labels
difficulty_mapping = {1: "Easy", 2: "Medium", 3: "Hard"}

# Convert numerical difficulty to categorical labels
df["difficulty_label"] = df["difficulty"].map(difficulty_mapping)

# Count occurrences of each difficulty level
difficulty_counts = df["difficulty_label"].value_counts()

# Ensure order of labels
name = ["Medium", "Easy", "Hard"]
values = [difficulty_counts.get("Medium", 0), difficulty_counts.get("Easy", 0), difficulty_counts.get("Hard", 0)]
colors = ["lightskyblue", "lightcoral", "lightgreen"]

# Plot the pie chart
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(values, labels=name, autopct="%0.2f%%", startangle=90, colors=colors)
ax.set_title("Hero Difficulty")

st.pyplot(fig)

