import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from text_utils import preprocess_text

class CivicTriageEngine:
    def __init__(self):
        print("Loading the RoBERTa & Scikit-Learn Assets")

        self.tokenizer = AutoTokenizer.from_pretrained("cardniffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

        