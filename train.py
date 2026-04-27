import pandas as pd
import joblib 
import os
import sys
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from text_utils import preprocess_text

#------------LOADING THE DATASET----------------

df = pd.read_csv("311_Service_Requests_for_2009.csv",low_memory=False)

cols = ['Complaint Type', 'Descriptor', 'Borough']

#------------HANDLING THE MISSING VALUES-----------

for col in cols:
    df[col] = df[col].fillna('Unknown')

#---------SAMPLING FOR SPEED SINCE IT'S A LARGE DATASET--------

df = df.sample(frac=1,random_state=42)
df.reset_index(drop=True,inplace=True)
df = df[0:20000]

df.info()
df['text'] = df['Complaint Type'] + ' ' + df['Descriptor']

#-------------APPLING THE TEXT PREPROCESSING----------------

df['clean_text'] = df['text'].apply(preprocess_text)
df[['text','clean_text']].head()


# ---------------NGRAMS--------------------------------------

#Unigram

vectorizer = CountVectorizer(ngram_range=(1,1), max_features=10)
X = vectorizer.fit_transform(df['clean_text'])

words = vectorizer.get_feature_names_out()
counts = X.sum(axis=0).A1

#bigram

vectorizer = CountVectorizer(ngram_range=(2,2), max_features=10)
X = vectorizer.fit_transform(df['clean_text'])

unigram_df = pd.DataFrame({'word': words, 'count': counts})
unigram_df.sort_values(by='count', ascending=False)

bigrams = vectorizer.get_feature_names_out()
counts = X.sum(axis=0).A1

bigram_df = pd.DataFrame({'bigram': bigrams, 'count': counts})
bigram_df.sort_values(by='count', ascending=False)

#-----------SPLITTING INTO INPUT AND TARGET VARIABLES---------

X = df['clean_text']
y = df['Complaint Type']

#----------------TFIDF VECTORIZER----------------------------

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(X)

#--------------TRAIN TEST SPLIT ------------------------------

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=100)

#--------------LOGISTIC REGRESSION----------------------------

from sklearn.model_selection import cross_val_score
# Create model
lr_model = LogisticRegression(max_iter=1000)

#------------TRAIN THE MODEL----------------------------------

lr_model.fit(X_train,y_train)
lr_model.score(X_train,y_train)

#------------CROSS VALIDATION---------------------------------

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(lr_model, X, y, cv=3, scoring='accuracy')

print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

print("\nCross Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())

lr_pred = lr_model.predict(X_test)

#Accuracy

print("Accuracy of Logistic Regression model is :", accuracy_score(y_test,lr_pred))
print("Classification Report of Logistic Regression model\n is :", classification_report(y_test,lr_pred))

if not os.path.exists('models'):
    os.makedirs('models')

#-------------SAVING THE MODELS------------------------------------

print('Saving the Logistic Regression model and Tfidf Vectorizer')
joblib.dump(lr_model, 'models/lr_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

if __name__ == "__main__":
    print("\n Training Completed. Models saved successfully!")

