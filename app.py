import streamlit as st
import joblib
import re
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import xgboost as xgb
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load models and transformers
models = {
    'XGBoost (Default)': 'best_xgb_model.pkl',
    'DecisionTreeClassifier': 'best_DecisionTreeClassifier_model.pkl',
    'AdaBoostClassifier': 'best_AdaBoostClassifier_model.pkl',
    'KNeighborsClassifier': 'best_KNeighborsClassifier_model.pkl',
    'LogisticRegression': 'best_LogisticRegression_model.pkl',
    'Naive Bayes': 'best_naive_bayes_model_model.pkl'
}

selected_model_name = st.sidebar.selectbox('Select Model', list(models.keys()), index=0)
selected_model_filename = models[selected_model_name]
model = joblib.load(selected_model_filename)

count = joblib.load('count_vectorizer.pkl')
tfidf = joblib.load('tfidf_transformer.pkl')

# Initialize stopwords and stemmer
stopword = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    words = text.split()
    filtered_words = [word for word in words if word not in stopword]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    cleaned_text = " ".join(stemmed_words)
    return cleaned_text


# App title and description
st.title('Offensive Language Classification App')
st.write("""
Determine if the input text contains offensive language using the selected model.
""")

# Sidebar with information and example
st.sidebar.title("About")
st.sidebar.info("""
This app uses various machine learning models to classify whether the input text contains offensive language.
Feel free to type in your text and click on the "Predict" button to see the prediction.
""")

st.sidebar.title("Interesting Examples")
st.sidebar.code("Example 1: 'That movie was a complete ass bitch disaster!' \nExample 2: 'The sunset was breathtaking.'", language="text")

# Input text box
input_text = st.text_area('Enter your text here:', '')

# Clean the input text
cleaned_input_text = clean_text(input_text)

# Transform the cleaned input text into TF-IDF features
input_text_vectorizer = count.transform([cleaned_input_text])
input_text_tfidf = tfidf.transform(input_text_vectorizer)

# Ensure the number of features matches the training data
if input_text_tfidf.shape[1] < count.transform(['']).shape[1]:
    input_text_tfidf = np.hstack((input_text_tfidf.toarray(), 
                                  np.zeros((input_text_tfidf.shape[0], 
                                            count.transform(['']).shape[1] - input_text_tfidf.shape[1]))))

# Make the prediction
if st.button('Predict'):
    prediction = model.predict(input_text_tfidf)
    result = "Offensive" if prediction[0] == 1 else "Not Offensive"
    
    st.write(f'Text Classification: {result} 🚫' if prediction[0] == 1 else f'Text Classification: {result} ✅')
