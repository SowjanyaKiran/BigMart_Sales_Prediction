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

    # Debugging: Print the raw input data
    st.write("Raw Input Data:")
    st.write(input_df)

    # Define the columns explicitly (categorical and numeric)
    categorical_cols = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
                        'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
    numeric_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']

    # Check if the input dataframe is empty or not
    if input_df.isnull().values.any():
        st.error("Input data contains missing values. Please fill in all fields.")
    else:
        # Handle missing values (fill NaN with a default value, e.g., mean for numeric, mode for categorical)
        input_df = input_df.apply(pd.to_numeric, errors='coerce')  # Coerce errors to NaN
        input_df.fillna(input_df.mean(), inplace=True)  # Fill NaN with column means for numeric columns

        # Debugging: Print the processed input data and types after handling NaN
        st.write("Processed Input Data (After NaN Handling):")
        st.write(input_df)
        st.write("Processed Data Types:", input_df.dtypes)

        # Label encoding for categorical columns only
        le = LabelEncoder()
        for col in categorical_cols:
            input_df[col] = le.fit_transform(input_df[col])

        # Debugging: Print the final processed input data and types
        st.write("Final Processed Input Data:")
        st.write(input_df)
        st.write("Final Processed Data Types:", input_df.dtypes)

        # Ensure that the model receives data in the same format and shape as the training data
        try:
            # Make prediction using the model
            prediction = model.predict(input_df)[0]  # Get the first element of the prediction
            st.success(f"📈 Predicted Item Outlet Sales: ₹{prediction:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
