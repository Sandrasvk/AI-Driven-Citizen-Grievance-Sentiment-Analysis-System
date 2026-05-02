Government & Public Sector: AI-Driven Citizen Grievance & Sentiment Analysis System

🚀 Project Overview
An automated system designed to triage NYC 311 Service Requests. The project transforms unstructured citizen complaints into actionable data by predicting the relevant government department, analyzing emotional sentiment, and calculating urgency scores in real-time — using a dual ML + AI pipeline

📂 Dataset

Dataset Name: NYC 311 Service Requests (2009)

Source: Kaggle

Link: https://www.kaggle.com/datasets/sheikmohamed/nyc-311-service-requests-for-2009/data

Main Source : https://data.cityofnewyork.us/Social-Services/311-Service-Requests-for-2009/3rfa-3xsf

📊 Dataset Details:

Original Size: ~1.7 million records
Sample Used: 20,000 records (for efficient processing)

 WORK PROGRESS

✅Phase 1: Data Engineering & EDA (Week 1)

Analyzed 20,000 records from the NYC 311 Service Request dataset
Implemented a custom NLP preprocessing pipeline:
Lowercase conversion
URL and special character removal
Stopword filtering
Lemmatization
Generated WordClouds and N-Gram (Unigram/Bigram) frequency analysis to identify top citizen concerns
Visualized Top 10 Complaint Categories, Borough Distribution, and Monthly Complaint Trends


✅ Phase 2: Machine Learning & Model Selection (Week 2)

Evaluated Random Forest vs. Logistic Regression for multi-class complaint classification
Applied TF-IDF Vectorization (unigram + bigram) to convert text into numerical feature vectors
Performed Cross-Validation using both cross_val_score and StratifiedKFold
Selected Logistic Regression for its balance of high accuracy and low latency in production
Generated Confusion Matrix and Classification Reports for both models
Exported trained artifacts:
models/lr_model.pkl
models/tfidf_vectorizer.pkl


✅ Phase 3: Deep Learning & Triage Logic (Week 3)

Migrated workflow from Jupyter Notebooks to production-grade Python scripts in VS Code
Integrated RoBERTa Transformer (cardiffnlp/twitter-roberta-base-sentiment) for sentiment detection
Classifies sentiment as: Positive, Neutral, Negative, Critical/Urgent
Developed a custom two-tier triage algorithm:
Tier 1: Strong critical keyword override (gas, fire, flooding → Emergency)
Tier 2: Sentiment-based urgency scoring with keyword boost
Assigns a mathematical priority score (urgency × 20, scale 0–100)
Performed batch processing on 2,000 records → saved as Week3_Final_Triage_Report.csv


✅ Phase 4: API Deployment (Week 4)

Built a high-performance REST API using FastAPI + Uvicorn
Implemented a dual inference pipeline per request:
🤖 RoBERTa → Sentiment + Urgency + Priority
📊 Logistic Regression → Department Routing
Developed a three-tier hybrid routing system:
Emergency override (gas leak → instant DEP routing)
Low confidence fallback (< 0.04 → Manual Review)
Validated ML routing via DEPT_MAP
Input validation with Pydantic BaseModel
Proper HTTP error handling (400, 422, 500)


🛠️ Tech Stack
Category
Technologies
Languages & Libraries
Python, Pandas, NumPy, Matplotlib, Seaborn
NLP
NLTK, spaCy concepts, WordCloud
AI/ML
Scikit-learn, HuggingFace Transformers (RoBERTa)
Web Framework
FastAPI, Uvicorn, Pydantic
Tools
VS Code, Git/GitHub, Jupyter Notebook


🌐 API Endpoints
GET /
Returns system status, version, and capabilities.
Response:
Json
​ {

  "status": "Online",
  
  "project": "AI-Driven Citizen Grievance System",
  
  "version": "1.0.0",
  
  "models_loaded": true,
  
  "capabilities": ["Sentiment Analysis", "Department Routing", "Priority Level", "Priority Score", "Confidence"]
}
 POST /triage
 
Accepts a raw citizen complaint and returns full triage analysis.

Request:

 {
 
  "text": "There is a gas leak in my building, it is very dangerous"
  
}

 Response
 
Json

{

  "status": "success",
  
  "metadata": {
  
    "timestamp": "2026-05-03 00:33:15",
    
    "ml_confidence": 1.0
    
  },
  "triage_results": {
  
    "original_input": "There is a gas leak in my building, it is very dangerous",
    
    "complaint_type": "Gas Leak (Emergency)",
    
    "assigned_agency": "Department of Environmental Protection",
    
    "sentiment": "Negative",
    
    "priority_level": "High",
    
    "priority_score": 80,
    
    "urgency_score": 4,
    
    "confidence": 0.911
  }
}
▶️ How to Run

1. Install Dependencies
   
Bash
pip install -r requirements.txt

 2. Train the Model
    
Bash
python train.py
 
 4. Start the API Server
    
Bash
uvicorn main:app --reload --port 8001

4. Access Swagger UI
   
Open in browser:

 http://127.0.0.1:8001/docs
