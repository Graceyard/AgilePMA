import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

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
github_url = "https://raw.githubusercontent.com/Graceyard/agilepma/refs/heads/master/cleaned_loan_data.csv"

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
    X = df.drop('Loan_Status', axis=1)
    X

    st.write('**y**')
    y = df.Loan_Status
    y
    
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

# Applicant Information (Input features)
with st.sidebar:
    st.header("Applicant Details")
    gender = st.selectbox("Gender", ["Female", "Male"])
    married = st.selectbox("Married Status", ["No", "Yes"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education Level", ["Undergraduate", "Graduate"])
    self_employed = st.selectbox("Self-Employed", ["No", "Yes"])

# Financial & Property Information
    st.header("Financial Details")
    applicant_income = st.text_input("Applicant Income ($)")
    coapplicant_income = st.text_input("Co-applicant Income ($)")
    loan_amount = st.slider("Loan Amount (in thousands)", min_value=1, max_value=500, step=1, value=100)
    loan_amount_term = st.selectbox("Loan_Amount_Term", ["12", "36", "60", "84", "120", "180", "240", "300", "360", "480"])
    credit_history = st.radio("Credit History Meets Guidelines?", ["Yes", "No"])
    property_area = st.radio("Property Area", ["Rural", "Semi-Urban", "Urban"])

# Create DF for input features 
    data = {'gender': gender,
            'married': married,
            'dependents': dependents,
            'education': education,
            'self_employed': self_employed,
            'applicant_income': applicant_income,
            'coapplicant_income': coapplicant_income,
            'loan_amount': loan_amount,
            'credit_history': credit_history,
            'property_area': property_area}
    input_df = pd.DataFrame(data, index=[0])
    input_loan = pd.concat([input_df, X], axis = 0)

input_loan

