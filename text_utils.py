import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

# Global objects
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --------------------------------------------------
# Text Preprocessing Function
# --------------------------------------------------
def preprocess_text(sentence):
    """
    Clean and preprocess text:
    - lowercase
    - remove urls
    - remove special chars:
    - remove stopwords
    - lemmatization
    """
    # Convert to lowercase
    text = str(sentence).lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Tokenize
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    words = [lemmatizer.lemmatize(word, "v") for word in words]
    return " ".join(words)

# --------------------------------------------------
# Apply preprocessing to list
# --------------------------------------------------
def preprocess_corpus(texts):
    """
    Apply preprocessing on list or pandas series
    """
    return[preprocess_text(text) for text in texts]