import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from sentiment_triage import CivicTriageEngine


app = FastAPI(title = "AI-Driven Citizen Grievance & Sentiment Analysis System", 
              description = """
              ### Automated 311 Triage Pipeline
              ***ML Engine:** Logistic Regression (Dept Routing)
              ***AI Engine:** RoBERTa (Sentiment, Priority & Urgency)
              """,
              version = "1.0.0"
              )

try:
    lr_model = joblib.load("models/lr_model.pkl")
    tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    triage_engine = CivicTriageEngine()
    print("SUCCESS: All model loaded successfully.")

except Exception as e:
    print(F"ERROR: Could not load assets. Check your file paths. Error:{e}")


class GrievanceRequest(BaseModel):
    text:str

@app.get("/")
def home():
        return{"status": "Online",
           "project": "AI-Driven Citizen Grievance System",
           "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "version": "1.0.0",
           "models_loaded": True,
           "capabilities": ["Sentiment Analysis", "Department Routing", "Priority Level", "Priority Score","Confidence"]
        }


@app.post("/triage")
async def process_grievance(item:GrievanceRequest):

        if not item.text.strip():
            raise HTTPException(status_code=400,detail="Text input is empty")
        
        try:

            analysis = triage_engine.analyze(item.text)

            vec = tfidf_vectorizer.transform([analysis['processed_text']])
            
            probs = lr_model.predict_proba(vec)[0]
            max_conf = max(probs)
            pred_label = lr_model.predict(vec)[0]
            
            DEPT_MAP = {
                "HEATING": "Department of Housing Preservation and Development",
                "HOT WATER": "Department of Housing Preservation and Development",
                "STREET LIGHT CONDITION": "Department of Transportation",
                "SEWER": "Department of Environmental Protection",
                "PLUMBING" : "Department of Environmental Protection",
                "GENERAL CONSTRUCTION" : "Department of Buildings",
                "PAINT-PLASTER": "Department of Housing Preservation and Development"
            }
            

            if max_conf < 0.25:
                 assigned_agency = "Manual review/ General City Triage"
                 final_complaint = "Unclassified Inquiry"
            else:
                 assigned_agency = DEPT_MAP.get(pred_label.upper(), "General City Triage")
                 final_complaint = pred_label.title()

            return{
                "status": "success",
                "metadata": {
                     "timestamp":
                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     "ml_confidence": round(float(max_conf),2)
                },
                

                "triage_results": {
                    "original_input": item.text,
                    "complaint_type": final_complaint,
                    "assigned_agency": assigned_agency,
                    "sentiment": analysis['sentiment'],
                    "priority_level": analysis['priority'],
                    "urgency_score": analysis['urgency_score'],
                    "priority_score": analysis.get('priority_score'),
                    "confidence" : analysis['confidence']
                }
            }
        
        
        except Exception as e:
            raise HTTPException(status_code=500, details=f"Inference Error: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port=8000)



