import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()


def tranform_text(text):
    text = text.lower()
    text = re.sub(re.compile(r'[^a-zA-Z\s]'), '', text) 
    text = word_tokenize(text)
    

    y = []
    for i in text:
        if i not in stopwords.words('english'):
           y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i)) 

    return ' '.join(y)
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/spam Classifier')

input_sms = st.text_input('Enter the message')

if st.button('predict'):

# 1- preprocess
    tranformed_sms = tranform_text(input_sms)
# 2- vectorize
    vector_input = tfidf.transform([tranformed_sms])
# 3- predict
    result = model.predict(vector_input)[0]
# 4- display
    if result == 1:
        st.header('spam')
    else:
        st.header('not spam')
