import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# load model
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

ps = PorterStemmer()

# preprocessing
def transform_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stopwords.words('english')
    ]

    return " ".join(words)

# UI
st.title("Spam Email Classifier")

input_sms = st.text_area("Enter Email")

if st.button("Predict"):

    transformed_sms = transform_text(input_sms)

    print(transformed_sms)

    vector_input = vectorizer.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    print(result)

    if result == 1:
        st.error("Spam")
    else:
        st.success("Not Spam")