import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import os
import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- STEP 1: PREPARATION ---
# Download tools so the script doesn't crash on a new computer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

df = pd.read_csv("311_Service_Requests_for_2009.csv",low_memory=False)
df = df.sample(frac=1,random_state=42)
df.reset_index(drop=True,inplace=True)
df = df[0:20000]

df = df[['Created Date','Complaint Type','Descriptor', 'Borough']]
df.info()
df['Complaint Type'] = df['Complaint Type'].fillna('Unknown')
df['Descriptor'] = df['Descriptor'].fillna('Unknown')
df['Borough'] = df['Borough'].fillna('Unknown')
df.isna().sum()
df['text'] = df['Complaint Type'] + ' ' + df['Descriptor']

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ----STEP  2: TEXT PREPROCESSING ----

def preprocess(sentence):
    text = str(sentence).lower()  # Convert to lowercase
    
    text = re.sub(r"http\S+|www\S+", " ", text) # Remove URLs
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) # Remove special characters

    words = text.split() 
    
# Remove stopwords + apply lemmatization
    
    words = [x for x in words if x not in stop_words]
    words = [lemmatizer.lemmatize(word, 'v') for word in words]

    return " ".join(words)

df['clean_text'] = df['text'].apply(preprocess)
df[['text','clean_text']].head()


# ----STEP 3 : Ngrams---------
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

# Splitting into input and target variables
X = df['clean_text']
y = df['Complaint Type']

#Tfidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(X)

#-----STEP 4:TRAIN TEST SPLIT ----------
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=100)

#----STEP 5: Logistic Regression model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Create model
lr_model = LogisticRegression(max_iter=1000)

# Train model
lr_model.fit(X_train,y_train)
lr_model.score(X_train,y_train)

#Cross validation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(lr_model, X, y, cv=3, scoring='accuracy')

print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

print("\nCross Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
lr_pred = lr_model.predict(X_test)

# Accuracy

print("Accuracy of Logistic Regression model is :", accuracy_score(y_test,lr_pred))
print("Classification Report of Logistic Regression model\n is :", classification_report(y_test,lr_pred))

if not os.path.exists('models'):
    os.makedirs('models')

#Saving the models
print('Saving the Logistic Regression model and Tfidf Vectorizer')
joblib.dump(lr_model, 'models/lr_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

if __name__ == "__main__":
    print("\n Training Completed. Models saved successfully!")

