import streamlit as st
import pickle
import nltk
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
from keras.preprocessing.sequence import pad_sequences


nltk.download('stopwords')

def cleanup_text(message):
    message = message.translate(str.maketrans('','',string.punctuation))
    words = [stemmer.stem(w) for w in message.split() if w.lower() not in stopwords.words('english') ]
    return ' '.join(words) 

def load_model(path='emailclass.h5'):
    return keras.models.load_model(path)

st.title('Email Categorization App')
with st.spinner('loading Spam classfication model'):
    model = load_model()
    vectorizer = load_model()
    #st.success('models loaded into memory')
    tokenizer=''
    with open('tokenizer.pickle','rb') as handle:
        tokenizer=pickle.load(handle)

message = st.text_area('Enter your email',value='Hi there')
btn = st.button('Process')



if btn and len(message)> 30:
    stemmer = SnowballStemmer('english')
    clean_msg = cleanup_text(message)
    
    vector = pad_sequences(tokenizer.texts_to_sequences([clean_msg]), maxlen=138)

    pred = model.predict(vector)


    x=0
    y=0
    for i in range(0,len(pred[0])):
        if y < round(pred[0][i],3):
            y = round(pred[0][i],3)
            x = i

    
    if x == 0:
        st.success("Spam Email")
    elif x == 1:
        st.success("Personal Email")
    elif x == 2:
        st.success("Professional Email")
    elif x == 3:
        st.success("Bills and reciept Email")
    elif x == 4:
        st.success("OTPs Email")
    elif x == 5:
        st.success("Shopping Email")
    else:
        st.error("Something is fishy in the Email")
      
    st.write("it is classified with ",round(pred[0][x]*100,2),"% probability.")
      
elif btn and len(message)<= 30:
    st.error("Email cannot be less than 30 characters.")