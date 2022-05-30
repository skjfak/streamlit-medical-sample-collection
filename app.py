import pickle
import streamlit as st
import numpy as np
import sklearn

data = pickle.load(open('final.pkl', 'rb'))
random_forest = pickle.load(open('rfd.pkl', 'rb'))

st.title("Medical Sample  Collection Streamline")

#Age
age = st.number_input('PATIENT AGE')

#cut-off time
time_cutt = st.number_input('Cut off time HH:MM')

#sample collection time
sc_time = st.number_input('TIME TAKEN FOR SAMPLE COLLECTION MM')

#lab location
lab_lo = np.log(st.number_input('LAB LOCATION IN KM'))

#Time Taken To Reach Lab MM
ttr_lab = np.log(st.number_input('TIME TAKEN TO REACH LAB MM'))

#gender
gender = st.radio('PATIENT GENDER', data['Patient_Gender'].unique())

#test name
tests = st.selectbox('TEST NAME', data['Test_Name'].unique())

#sample
sample = st.radio('SAMPLE TYPE', data['Sample'].unique())

#war of storage
storage = st.radio('STORAGE TYPE', data['Way_Of_Storage_Of_Sample'].unique())

#schedule
schedule = st.selectbox('CUT OFF SCHEDULE', data['Cut-off Schedule'].unique())

#traffic
traffic = st.selectbox('TRAFFIC CONDITION', data['Traffic_Conditions'].unique())

if st.button('Predict Result'):

    query = np.array([age, time_cutt, sc_time, lab_lo, ttr_lab, gender, tests, sample, storage, schedule, traffic])
    query = query.reshape(1,11)

    result = random_forest.predict(query)

    if result == 'Y':
        st.header("Sample Reached On Time ?\n YES")
    else:
        st.header("Sample Reached On Time ?\n NO")

