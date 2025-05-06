import pandas as pd
import numpy as np

from xgboost import XGBClassifier
import pickle
import joblib

import streamlit as st

#++++++++++#

#Judul
st.write("""
         <div style = "text-align: center;">
         <h2> Churn Customer Prediction </h2>
         </div>
         """, unsafe_allow_html=True)

#Sidebar menu imput
st.sidebar.header("Please Input Your Customer Feature")

    #untuk setiap inputan numerik
def User_input_feature():
    cred_score = st.sidebar.slider(label = 'Credit Score',min_value=350,max_value=850,value=500)
    Balance = st.sidebar.slider(label = 'Balance',min_value=0,max_value=251000,value=10000)
    Estimated_Salary = st.sidebar.slider(label = 'EstimatedSalary',min_value=11,max_value=200000,value=10000)

    Age = st.sidebar.number_input(label='Age',max_value=92,min_value=18,value = 40)
    Tenure = st.sidebar.number_input(label='Tenure',max_value=10,min_value=0)
    NumOfProduct = st.sidebar.number_input(label='Num of Product',max_value=5,min_value=1)

    #untuk setiap inputan kategorik

    CreditCard = st.sidebar.selectbox(label='Has Credit Card',options= [0,1])
    ActiveMember = st.sidebar.selectbox(label='is Active Member',options= [0,1])
    Gender = st.sidebar.selectbox(label='Gender',options= ["Female","Male"])
    Geography = st.sidebar.selectbox(label='Geography',options= ["France", "Germany", "Spain"])
    
    df = pd.DataFrame()
    df["CreditScore"]= [cred_score]
    df["Geography"] = [Geography]
    df["Gender"] = [Gender]
    df['Age']= [Age]
    df['Tenure'] = [Tenure]
    df['Balance'] = [Balance]
    df['NumOfProducts'] = [NumOfProduct]
    df['HasCrCard'] = [CreditCard]
    df['IsActiveMember'] = [ActiveMember]
    df['EstimatedSalary'] = [Estimated_Salary]

    return df
df_feature = User_input_feature()

#memamnggil model
model = joblib.load("model_Xgboost_joblib")

#predict
pred = model.predict(df_feature)


st.write("<b> Tujuan dari project ini adalah menentukan apakah seorang customer akan melakukan churn (tidak menggunakan jasa lagi) dari bank ini. </b>)", unsafe_allow_html=True)
#membuat layout menjadi 2 bagian
col1,col2 = st.columns(2)
with col1:
    st.subheader("Customer Characteristic")
    st.write(df_feature.transpose())

with col2:
    st.subheader("Predicted Result")
    if pred == 1:
        st.write('<h5 style="color: red;">Your Customer is likely to Churn</h1>', unsafe_allow_html=True)
    else:
        st.write('<h5 style="color: green;">Your Customer is likely to not Churn</h1>', unsafe_allow_html=True)