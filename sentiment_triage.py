import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from text_utils import preprocess_text

class CivicTriageEngine:
    def __init__(self):
        print("Loading the RoBERTa & Scikit-Learn Assets")

        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")


    def get_sentiment_probabilities(self,clean_text):
        inputs = self.tokenizer(clean_text,return_tensors='pt',truncation=True,max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits,dim=-1).numpy()[0]
        return probs
    

if __name__ == "__main__":
    triage = CivicTriageEngine()
    test_text = "street light conditon street light out"
    result = triage.get_sentiment_probabilities(test_text)
    print(f"Test result : {result}")


