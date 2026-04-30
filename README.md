Government & Public Sector: AI-Driven Citizen Grievance & Sentiment Analysis System

​🚀 Project Overview

​An automated system designed to triage NYC 311 Service Requests. The project transforms unstructured citizen complaints into actionable data by predicting the relevant department, analyzing emotional sentiment, and calculating urgency in real-time.

📂 Dataset

Dataset Name: NYC 311 Service Requests (2009)

Source: Kaggle

Link: https://www.kaggle.com/datasets/sheikmohamed/nyc-311-service-requests-for-2009/data

Main Source : https://data.cityofnewyork.us/Social-Services/311-Service-Requests-for-2009/3rfa-3xsf

📊 Dataset Details:

Original Size: ~1.7 million records
Sample Used: 20,000 records (for efficient processing)

 WORK PROGRESS

​Phase 1: Data Engineering & EDA (Week 1)

​Dataset: Analyzed 20,000+ records from the NYC 311 Service Request dataset.
​Preprocessing: Implemented a custom NLP pipeline including lowercase conversion, removal of URLs/special characters, stopword filtering, and Lemmatization.
​Insights: Generated WordClouds and N-Gram (Unigram/Bigram) analysis to identify top citizen concerns (Noise, Potholes, Utilities).


​Phase 2: Machine Learning & Model Selection (Week 2)
​Methodology: Evaluated Random Forest vs. Logistic Regression for classification.
​Feature Extraction: Utilized TF-IDF Vectorization to convert textual descriptions into numerical feature vectors.
​Optimization: Selected Logistic Regression for its balance of high accuracy and low latency in production environments.
​Artifacts: Exported trained model (lr_model.pkl) and vectorizer (tfidf_vectorizer.pkl) for API integration.

​Phase 3: Deep Learning & Triage Logic (Week 3)
​Evolution: Migrated workflow from Jupyter Notebooks to VS Code to implement production-grade Python scripts.
​Sentiment Engine: Integrated a RoBERTa-based Transformer model to detect citizen frustration levels.
​Priority Logic: Developed a custom triage algorithm that calculates Urgency Scores based on the intersection of complaint type and sentiment intensity.

​Phase 4: API Deployment (Week 4 - Current)
​Framework: Currently started to  build a high-performance REST API using FastAPI.

​🛠️ Tech Stack
​* Languages & Libraries : Python (Pandas, NumPy), Matplotlib, Seaborn

​* AI/ML: Scikit-learn, NLTK, HuggingFace Transformers (RoBERTa)

​* Web Framework: FastAPI, Uvicorn, Pydantic

​* Tools: VS Code, Git/GitHub, Jupyter Notebook
