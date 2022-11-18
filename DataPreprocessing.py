import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_score, precision_recall_curve, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import *

#Clean the dataset by removing no match rows
song_data = pd.read_csv("Labeled_Data.csv")
song_data = song_data[['artist','song','text','explicit_label']]
song_data = song_data.loc[song_data['explicit_label'] != 'no match']
#remove'\n' from the lyrics
re_drop = re.compile(r'\n')        
song_data[['text']] = song_data[['text']].applymap(lambda x:re_drop.sub(' ',x))


#Split into training dataset and test dataset
#extract all the rows with explicit_label = True
explicit = song_data.loc[song_data['explicit_label'] == 'True']

#ramdomly extract 1356 rows with explicit_label = False, which is the same as many as song_data_1
clean = song_data.loc[song_data['explicit_label'] == 'False']
clean = clean.sample(n=1356, replace=False, random_state=100)

X = explicit.iloc[:,:-1].append(clean.iloc[:,:-1])

Y = explicit.iloc[:,-1].append(clean.iloc[:,-1])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1356, random_state=100)

train_data = []
for i in range(len(X_train)):
    text = X_train.iloc[i,2]
    train_data.append(text)
    
test_data = []
for i in range(len(X_test)):
    text = X_test.iloc[i,2]
    test_data.append(text)
    
#Define Explicit to 1, Clean to 0
train_label = []
for i in range(len(Y_train)):
    o = Y_train.iloc[i]
    if o=="False":
        o = 0
    else:
        o = 1
    train_label.append(o)

test_label = []
for i in range(len(Y_test)):
    p = Y_test.iloc[i]
    if p=="False":
        p = 0
    else:
        p = 1
    test_label.append(p)
    
    
#Customized Features
#Import Bad Words List
file = open('Badwords.txt','r')
file = list(file)
bad_words = []
for w in file:
    bad_words.append(re.sub(r'\n','',w))
    
#Create customized features
def get_bad_words(review):
  target_word = bad_words
  count = 0
  threshold = 0
  for t in target_word:
        if review.find(t) != -1:
            count += 1
  return count > threshold

def get_num_words(review):
  threshold = 0
  words = str(review).split(' ')
  count = len(list(words))
  return count > threshold

def get_lda_words(review):
  target_word = ['chorus','girl','money','baby','nigga','bitch','want','love','wanna','gonna','come','right','shit','feel']
  count = 0
  threshold = 0
  for t in target_word:
        if review.find(t) != -1:
            count += 1
  return count > threshold

class CustomFeats(BaseEstimator, TransformerMixin):
    def __init__(self):
      self.feat_names = set()

    def fit(self, x, y=None):
        return self

    @staticmethod
    def features(review):
      return {
          'num_word': get_num_words(review),
          'bad_word': get_bad_words(review),
          'lda_word': get_lda_words(review)
      }

    def get_feature_names(self):
        return list(self.feat_names)
      
    def transform(self, reviews):
      feats = []
      for review in reviews:
        f = self.features(review)
        [self.feat_names.add(k) for k in f] 
        feats.append(f)
      return feats

#feats = make_pipeline(CustomFeats(), DictVectorizer())
feats = FeatureUnion([
     ('custom', make_pipeline(CustomFeats(), DictVectorizer())),
     ('bag_of_words', TfidfVectorizer(stop_words='english'))
 ])


#Data Modelling
def classification(feats, model):  
  train_vecs = feats.fit_transform(train_data)
  test_vecs = feats.transform(test_data)
    
  model.fit(train_vecs, train_label)

  train_preds = model.predict(train_vecs)
  train_f1 = f1_score(test_label, train_preds, average='macro')*100,'%'

  test_preds = model.predict(test_vecs)
  test_f1 = f1_score(test_label, test_preds, average='macro')*100,'%'

  cm = confusion_matrix(test_label, test_preds)
  print("Confusion Matrix : \n", cm, " \n")

  report = classification_report(test_label, test_preds)
  #print(report)

  return train_f1,test_f1


#Logistic Regression
model_lo= LogisticRegression(C=50)
classification(feats, model_lo)

#Random Forest
model_rf = RandomForestClassifier(n_estimators=110, max_depth=140, min_samples_split=30)
classification(feats, model_rf)

filename = 'finalized_model_rf.sav'
pickle.dump(model_rf, open(filename, 'wb'))

#KNN
model_knn= KNeighborsClassifier(n_neighbors=10) 
classification(feats, model_knn)

#Decision Tree
model_dt = DecisionTreeClassifier(min_samples_split=0.4, max_depth=77)
classification(feats, model_dt)

#SVM
model_svm = SVC(C = 10000, kernel = 'rbf')
classification(feats, model_svm)