import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import plotly.express as px

from PIL import Image


pickle_in = open("sarima_model.pkl","rb")
model=pickle.load(pickle_in)


def welcome():
    return "Welcome All"


def predict_note_authentication(data):

    prediction=model.forecast(data)
    #print(prediction)

    #Convert into a data frame
    output_df = pd.DataFrame(prediction)

    return output_df

def main():
    st.title("TS Authenticator")
    
    data = st.number_input('Enter the number of months to forecast', min_value=1, value=3)
    
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(data)
        st.dataframe(result)   # gave output in table
    
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
        st.text("Built by satwik")
    st.markdown('---')
    st.markdown('Developed by Satwik')
    

if __name__=='__main__':
    main()