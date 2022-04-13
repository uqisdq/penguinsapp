import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App

This App will help you predicts the **Palmer Penguin** Species

Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header("User Input Features")

st.sidebar.markdown("""
[Click Here for CSV Example input file Formatting](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# User's input parameters
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file below :",type=["csv"])

st.sidebar.write("""
*Or, if you don't want to create the input file, simply input the parameters below :*
""")
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_df['input'] = True
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
    input_df['input'] = True

penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins_raw['input'] = False
penguins = penguins_raw.drop(columns = ['species'])
df = pd.concat([input_df,penguins],axis=0)

encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col],prefix = col)
    df = pd.concat([df,dummy],axis = 1)
    del df[col]

df = df[df['input']==True]
df = df.drop(['input'],axis = 1)

st.subheader('User Input Features :')
if uploaded_file is not None :
    st.write(df)
else:
    st.write('Awaiting CSV File to be uploaded. Currently using the data from input sidebar (shown below) :')
    st.write(df)

#Load model
load_clf = pickle.load(open('penguins_clf.pkl','rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
