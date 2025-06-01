import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
raw_mail_data=pd.read_csv("D:\mlass2\mail_data.csv")
print(raw_mail_data.isna().sum())
mail_data=raw_mail_data.fillna('')
print(mail_data.shape)


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
mail_data['Category']=le.fit_transform(mail_data['Category'])
print(mail_data)


x=mail_data['Message']
y=mail_data['Category']
print(x,y)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=3)

print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)

feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)


xtrain_features=feature_extraction.fit_transform(xtrain)
xtest_features=feature_extraction.transform(xtest)

ytrain=ytrain.astype('int')
ytest=ytest.astype('int')

model=LogisticRegression()
model.fit(xtrain_features,ytrain)

prediction=model.predict(xtrain_features)
accuracyofmodel=accuracy_score(ytrain,prediction)

print(accuracyofmodel)

import pickle
with open('modelj.pkl', 'wb') as m:
    pickle.dump(model, m)

import streamlit as st
import joblib

# Load your trained model
with open('modelj.pkl', 'rb') as m:
    model = pickle.load(m)

st.title('Spam mail detection')
st.header('User Input')

import numpy as np

with open('modelv.pkl', 'wb') as f:
    pickle.dump(feature_extraction, f)
# Collect input from the user
input_text = st.text_input('Enter your mail')
with open('modelv.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
input_features = vectorizer.transform([input_text])






# Perform inference
if st.button('Predict'):
    prediction = model.predict(input_features)
    st.write('Predicted Class:', prediction)
    if prediction[0]==1:
        st.write("it is spam")
    else:
        st.write("it is not a spam")
