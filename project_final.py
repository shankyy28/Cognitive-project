import streamlit as st
import pandas as pd
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model, tokenizer, and label encoder
with open("E:\Code\Cognitive\model.pkl", 'rb') as file:
    model = pickle.load(file)

with open("E:\Code\Cognitive\tokenizer_model.pkl", 'rb') as file:
    tokenizer = pickle.load(file)

with open("E:\Code\Cognitive\le_model.pkl", 'rb') as file:
    le = pickle.load(file)

##############################################################################

stop_words = set(stopwords.words("english"))

def remove_stop_words(text):
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    Text = ''.join([i for i in text if not i.isdigit()])
    return Text

def lower_case(text):
    Text = [i.lower() for i in str(text).split()]
    return " ".join(Text)

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    Text = [lemmatizer.lemmatize(i) for i in str(text).split()]
    return " ".join(Text)

def mood(text):
    text = remove_stop_words(text)
    text = Removing_numbers(text)
    text = lower_case(text)
    text = lemmatization(text)
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen = 229, truncating='pre')
    result = le.inverse_transform(np.argmax(model.predict(text), axis = -1))
    proba =  np.max(model.predict(text))
    result = result[0]
    return result, proba

##############################################################################

st.title("Sentiment Analysis")

user_input = st.text_area("Enter your text:")

if st.button("Predict Sentiment"):
    if user_input:
        sentiment, score = mood(user_input)
        st.write("Predicted Sentiment:", sentiment)
        st.write("Confidence Scores:", score)
    else:
        st.write("Please enter some text.")