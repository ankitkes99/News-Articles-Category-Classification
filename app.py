from flask import Flask,request,render_template
import pickle
import spacy
import numpy as np
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re, string, unicodedata



app=Flask(__name__)
svc=pickle.load(open('final_model.pkl','rb'))

dictionary=pickle.load(open('final_dict.pkl','rb'))

#Text preprocessing Step

def remove_non_ascii(words):
    
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_non_alpha(words):
    
    new_words = []
    for word in words:
        new_word = re.sub(r'[^a-zA-Z]', "", word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
   
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

nlp = spacy.load('en', disable=['parser', 'ner'])
def lemmatize(clean_text):
   
    doc = nlp(clean_text)
    s=" ".join([token.lemma_ for token in doc])
    return s

 def clean_text(text):
  words = nltk.word_tokenize(text)

  words = remove_non_ascii(words)

  words = to_lowercase(words)

  words = remove_non_alpha(words)

  words = remove_stopwords(words)

  clean_text=" ".join([token for token in words])

  lemmatize_text=lemmatize(clean_text)

  return lemmatize_text


w2v_words= dict()
f = open('glove.6B.100d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    w2v_words[word] = coefs
f.close()

tfidf_feat=dictionary.keys()

id_to_category = {0:'Politics',1:'Technology', 2:'Entertainment',3: 'Business',4:'Sport'}


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    text=request.form['text']
    sent=clean_text(text)
    sent_vec = np.zeros(100)
    weight_sum = 0
    for word in sent.split():
        if word in w2v_words and word in tfidf_feat:
            vec = w2v_words[word]
            tf_idf = dictionary[word] * (sent.count(word) / len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum

    val=svc.predict(sent_vec.reshape(1,100))
    x=svc.predict_proba(sent_vec.reshape(1,100))
    dict_prob=dict()
    for idx in range(0,5):
        dict_prob[id_to_category[idx]]=round(x[0][idx]*100,2)
    return render_template('details.html',result=dict_prob,prediction_text=id_to_category[val[0]])

if __name__=='__main__':
    app.run(debug=True)
