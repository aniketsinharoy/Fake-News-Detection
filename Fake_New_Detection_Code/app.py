from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  
from nltk.stem.porter import PorterStemmer  
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
app=Flask(__name__)



def stemming(content):
    ps = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)       # Remove numbers and punctuations from content & replace them with an empty string
    stemmed_content = stemmed_content.lower()                # Convert everything to lowercase
    stemmed_content = stemmed_content.split()                # Create a list ["the", "is", ...]
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')] # Remove stopwords and perform stemming
    stemmed_content = ' '.join(stemmed_content)              # Join the list into a string
    return stemmed_content

@app.route('/')
def index_view():
    return render_template("index.html")
 
@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        with open("logisticRegressionSavedModelTitleAuthor.pkl", "rb") as file:
            loaded_model=pickle.load(file)
        vectorizer = TfidfVectorizer()

        t=request.form.get('title')
        a=request.form.get('author')

        text=t+' '+a
        
        new_text = stemming(text)
        with open('vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        new_data_vectors = vectorizer.transform([new_text])
        
        predict = loaded_model.predict(new_data_vectors)
        
        if int(predict)== 0:
            return render_template("real.html", prediction=predict)
        else:
            return render_template("fake.html", prediction=predict)
if __name__=="__main__":
    app.run()
