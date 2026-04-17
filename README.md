# 🏛️ Citizen Grievance Analysis using NLP

🚀 Internship Project (Week 1 & Week 2)

## 📌 Project Overview

This project focuses on analyzing citizen complaint data using Natural Language Processing (NLP) and basic Machine Learning techniques.

The work completed so far includes cleaning raw complaint text, exploring patterns in the data, and building a model to classify complaints into relevant departments.

## ❗ Problem Statement

Public grievance systems receive a large number of complaints in text format. Manually analyzing and categorizing these complaints is time-consuming and inefficient.

This project aims to build a system that can automatically process complaint text and classify it into appropriate departments, helping improve response time and service efficiency.

## 🎯 Objective

- Clean and preprocess complaint text  
- Perform Exploratory Data Analysis (EDA)  
- Extract useful text features  
- Build a classification model  

## 📊 Dataset

📂 Dataset

Dataset Name: NYC 311 Service Requests (2009)
Source: Kaggle
Link: https://www.kaggle.com/datasets/sheikmohamed/nyc-311-service-requests-for-2009/data
Main Source : https://data.cityofnewyork.us/Social-Services/311-Service-Requests-for-2009/3rfa-3xsf

📊 Dataset Details:

Original Size: ~1.7 million records
Sample Used: 20,000 records (for efficient processing)
📥 Dataset Access

The dataset is not included in this repository due to its large size and internship guidelines.

⚠️ Dataset is not included in this repository due to its large size.

### ▶️ How to Use
1. Download dataset from the link above  
2. Rename file to: 311 service requests for 2009.csv  
3. Place it in the project folder  

## 🧹 Week 1 – Data Cleaning & EDA

- Loaded dataset and selected relevant columns  
- Performed text preprocessing:
  - Lowercasing  
  - URL removal  
  - Special character removal  
  - Stopwords removal  
  - Lemmatization  
- Conducted EDA to understand complaint patterns  
- Created visualizations (WordCloud, plots)  
- Extracted features using unigram and bigram (CountVectorizer)  

## 🤖 Week 2 – Classification Model

- Converted text into numerical form using *TF-IDF*  
- Applied train-test split  
- Trained models:
  - Logistic Regression  
  - Random Forest
- Evaluated using:
  - Accuracy  
  - Confusion Matrix  
  - Cross-validation  

## 🧠 Workflow

Text Cleaning → EDA → Feature Extraction → Model Training → Evaluation  


## 🛠️ Tools Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- NLTK  
