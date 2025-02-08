import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier

# Streamlit Title
st.title('Loan Eligibility Checker 🏠💰🤝')
st.markdown("""
### 🔒 Internal Use Only  
This tool is designed for internal staff to assess loan eligibility based on predefined criteria.  
It does not guarantee final loan approval. Please use it for reference purposes only.""")

# Sidebar Title
st.sidebar.title("Loan Eligibility Checker")

# Sidebar: File Uploader function
uploaded_file = st.sidebar.file_uploader("Upload a CSV file 📤", type=["csv"])

# GitHub dataset URL 
github_url = "https://raw.githubusercontent.com/Graceyard/agilepma/refs/heads/master/Cleaned_loan_data%20copy.csv"

# Load data from either uploaded file or GitHub
if uploaded_file is not None:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    st.warning("📤 No file uploaded. Loading sample data from GitHub...")
    df = pd.read_csv(github_url)

# Display the uploaded file
with st.expander('Data'):
    st.write("#### Data Preview🔎:")
    st.dataframe(df.head())

    # X and Y
    st.write('**X**')
    X_raw = df.drop('Loan_Status', axis=1)
    X_raw

    st.write('**y**')
    y_raw = df.Loan_Status
    y_raw
    
# Distribution Visualization
with st.expander('Data Distribution Visualization'):
    st.write("#### Loan Status Distribution📊:")

# Check if 'Loan_Status' column exists
    if "Loan_Status" in df.columns:
        # Count approved (1) and rejected (0) loans
        status_counts = df["Loan_Status"].value_counts()

       # Set up the plot
        plt.figure(figsize=(8, 4))
        ax = sns.countplot(x='Loan_Status', data=df, palette="Set1")
        
        # Add bar labels to each bar
        for container in ax.containers:
            ax.bar_label(container)
        
        # Add title and labels
        plt.xlabel("Loan Status", fontsize=10)
        plt.xticks([0, 1], ['Rejected', 'Approved'], fontsize=10)
        plt.ylabel("Number of Applicants", fontsize=10)
        
        # Show the plot in Streamlit
        st.pyplot(plt)

    else:
        st.error("❌ The uploaded file does not contain a 'Loan_Status' column.")

#Input features 
# Applicant Information (Input features)
with st.sidebar:
    st.header("Applicant Details")
    gender = st.selectbox("Gender", ["Female", "Male"])
    married = st.selectbox("Married Status", ["No", "Yes"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education Level", ["Not Graduate", "Graduate"])
    self_employed = st.selectbox("Self-Employed", ["No", "Yes"])

# Financial & Property Information
    st.header("Financial Details")
    property_area = st.radio("Property Area", ["Rural", "Semiurban", "Urban"])
    applicant_income = st.text_input("Applicant Income ($)")
    coapplicant_income = st.text_input("Co-applicant Income ($)")
    loan_amount = st.slider("Loan Amount (in thousands)", min_value=1, max_value=500, step=1, value=100)
    loan_amount_term = st.selectbox("Loan_Amount_Term", ["12", "36", "60", "84", "120", "180", "240", "300", "360", "480"])
    credit_history = st.radio("Credit History Meets Guidelines?", ["0", "1"])

# Create DF for input features 
    data = {'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'Property_Area': property_area,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history}
    input_df = pd.DataFrame(data, index=[0])
    input_loan_status = pd.concat([input_df, X_raw], axis = 0)

with st.expander('Input Information'):
    st.write('**Input Information**')
    input_df
    st.write('**Loan data**')
    input_loan_status

# Data Preparation
# Define categorical columns to encode (X)
encode = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
df_loan = pd.get_dummies(input_loan_status[encode], prefix=encode)

# Add the numerical features back to the encoded DataFrame
df_loan = pd.concat([df_loan, input_loan_status.drop(columns=encode)], axis=1)

# Use iloc to slice
X = df_loan.iloc[1:]  # All rows except the first
input_row = df_loan.iloc[:1] # First row 

# Encode y
target_mapper = {'N':0,
                'Y':1}
#customize function
def target_encode(val):
    return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data Preparation'):
    st.write('**Encoded input loan (X)**')
    input_row
    st.write('**Encoded y**')
    y

# Ensure input_row alinged with the model input
model_columns = X.columns
input_row = pd.DataFrame(input_row, columns=model_columns)

# Use RF to train the classification model (refer Colab)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Apply model to make prediction 
prediction = model.predict(input_row)
prediction_proba = model.predict_proba(input_row)

prediction_proba
