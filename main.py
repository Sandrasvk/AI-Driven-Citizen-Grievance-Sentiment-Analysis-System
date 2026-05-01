import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
           "version": "1.0.0",
           "models_loaded": True
        }


@app.post("/triage")
async def process_grievance(item:GrievanceRequest):

        if not item.text.strip():
            raise HTTPException(status_code=400,detail="Text input is empty")
        
        try:

            analysis = triage_engine.analyze(item.text)

            vec = tfidf_vectorizer.transform([analysis['processed_text']])
            pred_label = lr_model.predict(vec)[0]
            
            DEPT_MAP = {
                "HEATING": "Department of Housing Preservation and Development",
                "HOT WATER": "Department of Housing Preservation and Development",
                "Street Light Condition": "Department of Health and Mental Hygiene",
                "Sewer": "Department of Environmental Protection"
            }
            
            agency = DEPT_MAP.get(pred_label,"General City Triage")

            return{
                "status": "success",
                "data": {
                    "complaint_type": pred_label,
                    "assigned_agency": agency,
                    "analysis":{
                        "sentiment": analysis['sentiment'],
                        "priority_level": analysis['priority'],
                        "urgency_score": analysis['urgency_score'],
                        "priority_score": analysis.get('priority_score'),
                        "confidence" : round(analysis['confidence'])
                    }
                }
            }
        
        except Exception as e:
            raise HTTPException(status_code=500, details=f"Inference Error: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port=8000)



