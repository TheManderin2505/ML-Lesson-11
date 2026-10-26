#nlp    natural language prossesing

import pandas as pd

from sklearn.model_selection import train_test_split

data = pd.read_csv("train.txt",sep=";",names=["text","label"],nrows=5000)

print(data.head(),data.info())

data["label"] = data["label"].replace(
    {"sadness":1,"anger":1,"fear":1,"joy":0,"love":0,"suprise":0}
)


print(data.head())

x= data["text"]
y = data["label"]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.5,random_state=1)

import nltk 
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re

wne = WordNetLemmatizer()

def transform(data):
    corpus = []
    for i in  data:
        newi = re.sub("[^a-zA-Z]"," ",i)
        newi = newi.lower()

        newi = newi.split()

        list1 = [wne.lemmatize(word) for word in newi if word not in stopwords.words("english")]
        corpus.append(" ".join(list1))
    return corpus

xtrain_corpus = transform(xtrain)
xtest_corpus = transform(xtest)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(1,2))     #bigram representation
xtrain_new = cv.fit_transform(xtrain_corpus)
xtest_new = cv.transform(xtest_corpus)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(xtrain_new,ytrain)

trainpred = rfc.predict(xtrain_new)
testpred = rfc.predict(xtest_new)

def ns(text):
    tr_text = transform(text)
    tr_text = cv.transform(tr_text)
    textpred = rfc.predict(tr_text)
    if textpred[0] == 1:
        print("negetive")
    else:
        print("positive")

text1 = ["today was a good day"]
text2 = ["today was a terible day, as it rained"]

print("sentiment prediction")
ns(text1)
ns(text2)

