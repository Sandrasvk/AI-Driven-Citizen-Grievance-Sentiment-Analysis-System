import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from sentiment_triage import CivicTriageEngine


#---------------------------APPLICATION METADATA--------------------------------------------------

app = FastAPI(title = "AI-Driven Citizen Grievance & Sentiment Analysis System", 
              description = """
              ### Automated 311 Triage Pipeline
              - **ML Engine:** Logistic Regression (Dept Routing)
              - **AI Engine:** RoBERTa (Sentiment, Priority & Urgency)
              """,
              version = "1.0.0"
              )


#---------------------------MODEL LOADING--------------------------------------------------------

try:
    lr_model = joblib.load("models/lr_model.pkl")
    tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    triage_engine = CivicTriageEngine()
    print("SUCCESS: All model loaded successfully.")

except Exception as e:
    print(f"ERROR: Could not load assets. Check your file paths. Error:{e}")


#---------------------------DEPARTMENT ROUTING MAP-------------------------------------------------

DEPT_MAP = {
    "HEAT/HOT WATER":        "Department of Housing Preservation and Development",
    "HEATING":               "Department of Housing Preservation and Development",
    "PAINT - PLASTER":       "Department of Housing Preservation and Development",
    "STREET LIGHT CONDITION":"Department of Transportation",
    "STREET CONDITION":      "Department of Transportation",
    "SEWER":                 "Department of Environmental Protection",
    "WATER SYSTEM":          "Department of Environmental Protection",
    "PLUMBING":              "Department of Environmental Protection",
    "GENERAL CONSTRUCTION":  "Department of Buildings",
    "UNSANITARY CONDITION":  "Department of Health and Mental Hygiene",
    "DAMAGED TREE":          "Department of Parks and Recreation"
}

#-------------------------------- REQUEST SCHEMA---------------------------------------------------

class GrievanceRequest(BaseModel):
    text:str


#--------------------------------HOME ENDPOINT--------------------------------------------------------
    
@app.get("/")
def home():
        return{"status": "Online",
           "project": "AI-Driven Citizen Grievance System",
           "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "version": "1.0.0",
           "models_loaded": True,
           "capabilities": [
               "Sentiment Analysis", 
               "Department Routing",
                "Priority Level",
                "Priority Score",
                "Confidence"
                ]                        
        }


#----------------------------------TRIAGE ENDPOINT-----------------------------------------------------

@app.post("/triage")
async def process_grievance(item: GrievanceRequest):
    # 1. Validation: Ensure the input isn't empty

    if not item.text.strip():
        raise HTTPException(status_code=400, detail="Text input is empty")
    
    try:
        # AL & ML Inference - ROBERTa + Logistic Regression together
        analysis = triage_engine.analyze(item.text)
        vec = tfidf_vectorizer.transform([analysis['processed_text']])
        probs = lr_model.predict_proba(vec)[0]
        max_conf = max(probs)
        pred_label = lr_model.predict(vec)[0]
        clean_label = str(pred_label).strip().upper()

        # 4. Hybrid Routing Logic
        text_up = item.text.upper()

        if "GAS" in text_up and "LEAK" in text_up:
            assigned_agency = "Department of Environmental Protection"
            final_complaint = "Gas Leak (Emergency)"
            max_conf = 1.0
        
        elif max_conf < 0.04:  
            assigned_agency = "Manual review/ General City Triage"
            final_complaint = "Unclassified Inquiry"
            
        else:
            assigned_agency = DEPT_MAP.get(clean_label, "General City Triage")
            final_complaint = clean_label.title()

        # JSON Response
        return {
            "status": "success",
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ml_confidence": round(float(max_conf), 2)
            },
            "triage_results": {
                "original_input": item.text,
                "complaint_type": final_complaint,
                "assigned_agency": assigned_agency,
                "sentiment": analysis.get('sentiment'),
                "priority_level": analysis.get('priority'),
                "priority_score": analysis.get('priority_score'),
                "urgency_score": analysis.get('urgency_score'),
                "confidence": analysis.get('confidence')
            }
        }

    except Exception as e:
        print(f"PIPELINE ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")
    
    
# Run Server
if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=8001)