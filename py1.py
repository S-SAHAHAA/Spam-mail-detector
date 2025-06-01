import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
st.title("spam mail detector")
model_loaded=pickle.load(open('D:\890\model_jas','rb'))

feature_extraction= TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)

def predict():
    input_mail=st.text_input("enter your mail:")
    input_features=feature_extraction.transform(input_mail)
    prediction=model_loaded.predict()
    if prediction==1:
        st.write_stream("it is spam")
    else:
        st.write_stream("it is not spam")



input_feature=st.text_input("Enter your mail")
confirm=st.button("check",on_click=predict)


