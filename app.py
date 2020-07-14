from flask import Flask,request,render_template
import pickle
from gensim import utils
import gensim.parsing.preprocessing as gsp
import spacy
import numpy as np



app=Flask(__name__)
svc=pickle.load(open('final_model.pkl','rb'))

dictionary=pickle.load(open('final_dict.pkl','rb'))

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

filters = [
           gsp.strip_tags,
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords,
           gsp.strip_short
          ]


def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
        doc = nlp(s)
        s = " ".join([token.lemma_ for token in doc])
    return s

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
