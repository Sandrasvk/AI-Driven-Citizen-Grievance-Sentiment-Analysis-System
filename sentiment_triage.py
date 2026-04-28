import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from text_utils import preprocess_text

#----------------------CORE ENGINE CLASS FOR SENTIMENT, PRIORITY AND URGENCY-----------------------------

class CivicTriageEngine:
    def __init__(self):

        # Initialize a ROBERTa Transformer Model
        print("Loading the RoBERTa & Scikit-Learn Assets") 

        self.tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        )

        # Sentiment labels
        self.labels = ["Negative", "Neutral", "Positive"]


    # -------------------SENTIMENT PROBABILITIES------------------------------------------------

    def get_sentiment_probabilities(self, clean_text):

        inputs = self.tokenizer(clean_text, return_tensors='pt', truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]

        return probs


    # ------------------SENTIMENT PREDICTION----------------------------------------------------
    def predict_sentiment(self,text):

        # preprocessing
        clean_text = preprocess_text(text)
        probs = self.get_sentiment_probabilities(clean_text)

        idx = np.argmax(probs)
        sentiment = self.labels[idx]
        confidence = float(probs[idx])

        return sentiment, confidence, clean_text


    # ------------------URGENCY----------------------------------------------------------------
    def compute_urgency(self, sentiment, clean_text):
     base_score = {
        "Positive": 1,
        "Neutral": 2,
        "Negative": 3
   }

     urgent_keywords = [
        "urgent", "asap", "immediately", "broken", "unsafe",
        "not working", "error", "failed", "complaint", "issue", "no water", "no heat"
    ]

     strong_critical = [
         "danger", "emergency", "gas", "leak", "fire", "explosion", "live wire", "flooding"
    ]


    # 🔥 Strong override (makes it high priority)
     for word in strong_critical:
        if word in clean_text:
            return 4
        
        boost = 0

    # Normal keyword boost
     for word in urgent_keywords:
        if word in clean_text:
            boost = 1
            break

     score = base_score[sentiment] + boost
     return min(score, 5)


    # -------------------------PRIORITY---------------------------------------------------------------
    def get_priority(self, score):
        if score >= 4:
            return "High"
        elif score == 3:
            return "Medium"
        else:
            return "Low"


    # --------------------FULL PIPLINE--------------------------------------------------------
    def analyze(self, text):
        sentiment, confidence, clean_text = self.predict_sentiment(text)
        urgency = self.compute_urgency(sentiment, clean_text)
        priority = self.get_priority(urgency)

        return {
            "original_text": text,
            "processed_text" : clean_text,
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "urgency_score": urgency,
            "priority": priority,
            "priority_score" : urgency * 20
        }


# ----------------MAIN EXECUTION-------------------------------------------

if __name__ == "__main__":
    triage = CivicTriageEngine()

    # Sample Testing
    sample1 = "URGENT: THE GAS LEAK IS DANGEROUS"
    result = triage.analyze(sample1)


    print("\n-----------FINAL TRIAGE RESULT--------------------")
    print(f"Original text of sample1 :{result['original_text']}")
    print(f"Processed text of sample1 :{result['processed_text']}")

    print(f"Sentiment of sample1 :{result['sentiment']}")
    print(f"Confidence of sample1: {result['confidence']}")
    print(f"Priority of sample1: {result['priority']}")
    print(f"(Urgency Score of sample1: {result['urgency_score']})")
    print(f"The mathematical priority score of sample1: {result['priority_score']}")

    sample2 = "The street light outside my house is not working and area feels dangerous at night"
    result_sample2 = triage.analyze(sample2)

    print(f"Original text of sample2 :{result_sample2['original_text']}")
    print(f"Processed text of sample2 :{result_sample2['processed_text']}")

    print(f"Sentiment of sample2 :{result_sample2['sentiment']}")
    print(f"Confidence of sample2: {result_sample2['confidence']}")
    print(f"Priority of sample2: {result_sample2['priority']}")
    print(f"(Urgency Score of sample2: {result_sample2['urgency_score']})")
    print(f"The mathematical priority score of sample2: {result_sample2['priority_score']}")


    #--------------311 DATASET BATCH PROCESSING--------------------------
    print("\nStarting Batch Processing on 2,000 rows......")

    # Checking sentiment,priority and urgency in our original dataset
    try:
        df = pd.read_csv("311_service_Requests_for_2009.csv",low_memory=False)
        batch = df.head(2000).copy()

        batch['input'] = batch['Complaint Type'].fillna("") + " " + batch['Descriptor'].fillna("")

        result = batch['input'].apply(triage.analyze)

        results_df = pd.DataFrame(list(result))
        final_report = pd.concat([batch,results_df],axis=1)

        final_report.to_csv("Week3_Final_Triage_Report.csv")
        print(f"SUCCESS: Report saved to 'Week3_Final_Triage_Report.csv")

    except Exception as e:

        print(f"Data process error: {e}")

