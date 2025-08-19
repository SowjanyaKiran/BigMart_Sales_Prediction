# 🛒 BigMart Sales Prediction using Machine Learning  

This project focuses on predicting **retail product sales** in BigMart outlets using a **Machine Learning pipeline** trained on historical sales data.  
The model takes various product and outlet attributes as input (like Item Weight, MRP, Fat Content, Outlet Type, etc.) and predicts the expected sales for that item.  


📂 **Dataset**  
The dataset contains details of products and outlets including:  
- Item Identifier, Weight, Fat Content, Visibility  
- Item Type, MRP (Maximum Retail Price)  
- Outlet Identifier, Size, Location Type, Type, and Age  
- Target Variable: **Item Outlet Sales**  

This structured data enables feature engineering and training of regression models.  

🛠️ **Project Workflow**  

📌 **1. Install Dependencies**  

pip install -r requirements.txt

📌 **2. Load Model**

Pre-trained model (bigmart_best_model.joblib) is loaded using joblib.
Metadata file (model_meta.json) is used to check scikit-learn version compatibility.

📌 **3. Streamlit Web App**

User-friendly form to input product & outlet details.
Handles both categorical and numerical features.
On clicking Predict, the app outputs estimated sales instantly.

📌 **4. Prediction**

Converts user inputs into a Pandas DataFrame.
Passes data through the trained ML pipeline.
Displays predicted sales value in INR 💰.

🧠 **Model Summary**

Preprocessing: Encoding categorical variables, scaling numeric ones.
Algorithms explored: Linear Regression, Random Forest, XGBoost, etc.
Final deployed model: Best-performing pipeline stored in .joblib.

📊 **Sample Inference**
Input:

Item MRP → 150.0
Fat Content → Low Fat
Outlet Identifier → OUT027
Outlet Size → Medium
Outlet Age → 15

**Output:**

📈 Predicted Item Outlet Sales: ₹2012.45


🖼️ **Architecture Diagram**

Below is the high-level flow of how this project works:
Explanation:
User Input (Streamlit UI) → Enter item & outlet details
Data Preprocessing → Encoding + Scaling
Trained ML Model → Predicts sales
Streamlit Output → Displays prediction

💻 **Tech Stack**

Python
Streamlit (web app framework)
Pandas (data handling)
Scikit-learn (ML pipeline & model)
Joblib (model serialization)

🙌 **Future Improvements**

Deploy app on Streamlit Cloud / AWS / Heroku.
Add data visualization dashboards for sales insights.
Explore deep learning models for advanced regression tasks.

📁 **License**
This project is for educational purposes only. Dataset is publicly available under its respective license.
