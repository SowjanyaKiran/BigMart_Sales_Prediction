import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load Model (unpack the tuple) from Pickle
with open("bigmart_best_model.pkl", "rb") as f:
    model, sklearn_version = pickle.load(f)  # Unpack the tuple (model, version)

# Set up the Streamlit app title and description
st.title("🛒 BigMart Sales Prediction App")
st.markdown(f"Using **scikit-learn v{sklearn_version}** model to predict item sales.")

# User Inputs
Item_Identifier = st.text_input("Item Identifier", "FDA15")
Item_Weight = st.number_input("Item Weight", min_value=0.0, value=12.5)
Item_Fat_Content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
Item_Visibility = st.slider("Item Visibility", min_value=0.0, max_value=0.3, step=0.01, value=0.1)
Item_Type = st.selectbox("Item Type", [
    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
    "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast",
    "Health and Hygiene", "Hard Drinks", "Canned", "Breads",
    "Starchy Foods", "Others", "Seafood"
])
Item_MRP = st.number_input("Item MRP", min_value=0.0, value=150.0)
Outlet_Identifier = st.selectbox("Outlet Identifier", [
    "OUT027", "OUT013", "OUT049", "OUT035", "OUT046",
    "OUT017", "OUT045", "OUT018", "OUT019", "OUT010"
])
Outlet_Size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
Outlet_Location_Type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
Outlet_Type = st.selectbox("Outlet Type", [
    "Supermarket Type1", "Supermarket Type2",
    "Supermarket Type3", "Grocery Store"
])
Outlet_Age = st.slider("Outlet Age (Years)", 0, 40, 15)

# Predict Button
if st.button("Predict Sales"):
    # Create a DataFrame for the input data
    input_df = pd.DataFrame([{
        "Item_Identifier": Item_Identifier,
        "Item_Weight": Item_Weight,
        "Item_Fat_Content": Item_Fat_Content,
        "Item_Visibility": Item_Visibility,
        "Item_Type": Item_Type,
        "Item_MRP": Item_MRP,
        "Outlet_Identifier": Outlet_Identifier,
        "Outlet_Size": Outlet_Size,
        "Outlet_Location_Type": Outlet_Location_Type,
        "Outlet_Type": Outlet_Type,
        "Outlet_Age": Outlet_Age
    }])

    # Handle missing values (fill NaN with a default value, e.g., mean for numeric, mode for categorical)
    input_df = input_df.apply(pd.to_numeric, errors='coerce')  # Coerce errors to NaN
    input_df.fillna(input_df.mean(), inplace=True)  # Fill NaN with column means for numeric columns

    # Handle necessary preprocessing (e.g., encoding categorical variables)
    le = LabelEncoder()
    input_df["Item_Fat_Content"] = le.fit_transform(input_df["Item_Fat_Content"])
    input_df["Item_Type"] = le.fit_transform(input_df["Item_Type"])
    input_df["Outlet_Identifier"] = le.fit_transform(input_df["Outlet_Identifier"])
    input_df["Outlet_Size"] = le.fit_transform(input_df["Outlet_Size"])
    input_df["Outlet_Location_Type"] = le.fit_transform(input_df["Outlet_Location_Type"])
    input_df["Outlet_Type"] = le.fit_transform(input_df["Outlet_Type"])

    # Ensure that the model receives data in the same format and shape as the training data
    try:
        # Make prediction using the model
        prediction = model.predict(input_df)[0]  # Get the first element of the prediction
        st.success(f"📈 Predicted Item Outlet Sales: ₹{prediction:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
