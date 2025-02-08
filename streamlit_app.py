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

if uploaded_file is not None:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)

with st.expander('Data'):
    st.write('**data**')
    df1 = pd.read_csv('https://github.com/Graceyard/agilepma/blob/master/cleaned_loan_data.csv')

# Display the uploaded file
    st.write("#### Uploaded Data Preview:")
    st.dataframe(df.head())

# Distribution Visualization
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

else:
    st.info("📤 Please upload a CSV file to proceed.")

# Applicant Information
st.sidebar.header("Applicant Details")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
married = st.sidebar.selectbox("Married Status", ["No", "Yes"])
dependents = st.sidebar.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education Level", ["Undergraduate", "Graduate"])
self_employed = st.sidebar.selectbox("Self-Employed", ["No", "Yes"])

# Financial & Property Information
st.sidebar.header("Financial Details")
applicant_income = st.sidebar.text_input("Applicant Income ($)")
coapplicant_income = st.sidebar.text_input("Co-applicant Income ($)")
loan_amount = st.sidebar.slider("Loan Amount (in thousands)", min_value=1, max_value=500, step=1, value=100)
loan_amount_term = st.sidebar.selectbox("Loan_Amount_Term", ["12", "36", "60", "84", "120", "180", "240", "300", "360", "480"])
credit_history = st.sidebar.radio("Credit History Meets Guidelines?", ["Yes", "No"])
property_area = st.sidebar.radio("Property Area", ["Rural", "Semi-Urban", "Urban"])
