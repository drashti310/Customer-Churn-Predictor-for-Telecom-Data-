# import required libraries

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

def main():

    # load the saved model from disk

    rf_model = pickle.load(open(r'model.pkl', 'rb'))

    # setting Application title

    st.title('Customer Churn Predictor')

    # setting Application description

    st.markdown(""":dart:  This web based application is made to predict customer churn in a ficitional telecommunication use case.""")
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    st.info("Input data below")

    # based on our optimal features selection defining input parameters

    st.subheader("Demographic data")
    Gender= st.selectbox('Gender:', ('Male','Female'))
    Dependents = st.selectbox('Dependents:', ('Yes', 'No'))
    Partner = st.selectbox('Partner:', ('Yes', 'No'))


    st.subheader("Payment data")
    Tenure_Months = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
    Contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    Paperless_Billing = st.selectbox('Paperless Billing', ('Yes', 'No'))
    Payment_Method = st.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer','Credit card'))
    Monthly_Charges = st.number_input('The amount charged to the customer monthly', min_value=0.00, max_value=150.00, value=0.00)
    Total_Charges = st.number_input('The total amount charged to the customer',min_value=0.00, max_value=10000.00, value=0.00)

    st.subheader("Services signed up for")
    Multiple_Lines = st.selectbox("Does the customer have multiple lines",('Yes','No','No phone service'))
    Phone_Service = st.selectbox('Phone Service:', ('Yes', 'No'))
    Internet_Service = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'))
    Online_Security = st.selectbox("Does the customer have online security",('Yes','No','No internet service'))
    Online_Backup = st.selectbox("Does the customer have online backup",('Yes','No','No internet service'))
    Device_Protection = st.selectbox("Does the custimer have Device Protection",('Yes','No','No internet service'))
    Tech_Support = st.selectbox("Does the customer have technology support", ('Yes','No','No internet service'))
    Streaming_TV = st.selectbox("Does the customer stream TV", ('Yes','No','No internet service'))
    Streaming_Movies = st.selectbox("Does the customer stream movies", ('Yes','No','No internet service'))

    # defining data dictionary with all variables

    data = {
                'Gender': Gender,
                'Partner': Partner,
                'Dependents': Dependents,   
                'Tenure_Months':Tenure_Months,
                'Phone_Service': Phone_Service,
                'Paperless_Billing': Paperless_Billing,
                'Monthly_Charges': Monthly_Charges,
                'Total_Charges': Total_Charges,
                'Multiple_Lines': Multiple_Lines,
                'Internet_Service': Internet_Service,
                'Online_Security': Online_Security,
                'Online_Backup': Online_Backup,
                'Device_Protection': Device_Protection,
                'Tech_Support': Tech_Support,
                'Streaming_TV': Streaming_TV,
                'Streaming_Movies': Streaming_Movies,
                'Contract': Contract,
                'Payment_Method':Payment_Method,  
                }

    # converting dictionary to dataframe

    features_df = pd.DataFrame.from_dict([data])

    # defining data preprocessing function

    def preprocess(df):

        # encoding gender category

        df['Gender'] = df['Gender'].map({'Female':1, 'Male':0})

        def binary_map(feature):
            return feature.map({'Yes':1, 'No':0})

        # encoding other binary category

        binary_list = ['Partner', 'Dependents', 'Phone_Service', 'Paperless_Billing']
        df[binary_list] = df[binary_list].apply(binary_map)

        # converting Monthly charges and Total charges columns into float datatype

        df['Monthly_Charges'] = df['Monthly_Charges'].astype(float)
        df['Total_Charges'] = df['Total_Charges'].astype(float)

        columns = ['Gender','Partner','Dependents', 'Tenure_Months', 'Phone_Service', 'Paperless_Billing', 'Monthly_Charges', 'Total_Charges', 'Multiple_Lines_No', 'Multiple_Lines_No phone service', 'Multiple_Lines_Yes', 'Internet_Service_DSL', 'Internet_Service_Fiber optic', 'Internet_Service_No', 'Online_Security_No', 'Online_Security_No internet service', 'Online_Security_Yes', 'Online_Backup_No', 'Online_Backup_No internet service', 'Online_Backup_Yes', 'Device_Protection_No', 'Device_Protection_No internet service', 'Device_Protection_Yes', 'Tech_Support_No', 'Tech_Support_No internet service', 'Tech_Support_Yes', 'Streaming_TV_No', 'Streaming_TV_No internet service', 'Streaming_TV_Yes', 'Streaming_Movies_No', 'Streaming_Movies_No internet service', 'Streaming_Movies_Yes', 'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year', 'Payment_Method_Bank transfer', 'Payment_Method_Credit card', 'Payment_Method_Electronic check', 'Payment_Method_Mailed check']
        
        # encoding the other categorical categoric features with more than two categories

        df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)

        # feature scaling

        sc = MinMaxScaler()
        df['Tenure_Months'] = sc.fit_transform(df[['Tenure_Months']])
        df['Monthly_Charges'] = sc.fit_transform(df[['Monthly_Charges']])
        df['Total_Charges'] = sc.fit_transform(df[['Total_Charges']])

        # return preprocessed input data

        return df 

    preprocess_df = preprocess(features_df)

    # using saved model predict preprocessed input data

    prediction = rf_model.predict(preprocess_df)

    # check probability of predicted data

    probability = rf_model.predict_proba(preprocess_df)
    yes_probability = (probability[-1,-1])*100
    no_probability = (probability[-1,1])*100

    html_str_yes = f"""
                <style>
                p.a {{
                font: bold 10 px Courier;
                color: red;
                }}
                </style>
                <p class="a">Chances of customer to churn is:  {yes_probability} %</p>
                """
    html_str_no = f"""
                <style>
                p.a {{
                font: bold 10 px Courier;
                color: green;
                }}
                </style>
                <p class="a">Chances of customer to churn is:  {no_probability} %</p>
                """

    # print the results based on prediction output

    if st.button('Predict'):

            if prediction == 1:
                st.warning('Yes, the customer will terminate the service.')
                st.markdown(html_str_yes, unsafe_allow_html=True)
            else:
                st.success('No, the customer is happy with Telco Services.')
                st.markdown(html_str_no, unsafe_allow_html=True)

if __name__ == '__main__':
        main()