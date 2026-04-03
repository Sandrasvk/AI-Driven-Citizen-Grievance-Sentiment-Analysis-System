Government & public Sector -AI-Driven Citizen Grievance & Sentiment Analysis System

🚧 Project Status

Ongoing Internship Project – Week 1 Completed

This project in week1 is part of an internship program focused on analyzing large-scale service request data using Natural Language Processing (NLP),Exploratory Data Analysis (EDA) etc .

🎯 Objective

The objective of this project is to analyze 311 service request complaints and extract meaningful insights from unstructured textual data using NLP techniques.

📂 Dataset

- Dataset Name: NYC 311 Service Requests (2009)
- Source: Kaggle
- Link: https://www.kaggle.com/datasets/sheikmohamed/nyc-311-service-requests-for-2009/data
- Main Source : https://data.cityofnewyork.us/Social-Services/311-Service-Requests-for-2009/3rfa-3xsf]

📊 Dataset Details:

- Original Size: ~1.7 million records
- Sample Used: 20,000 records (for efficient processing)

📥 Dataset Access

The dataset is not included in this repository due to its large size and internship guidelines.

To run this project:

1. Download the dataset from the Kaggle link above
2. Ensure the file name is:
   "311 service requests for 2009.csv"
3. Place the file in the same folder as the notebook


🧹 Work Completed (Week 1)

✔ Data Loading & Inspection

- Loaded dataset using Pandas
- Checked dataset shape and structure
- Sampled,Slicing done for select 20,000 rows efficiently

✔ Data Cleaning

- Handled missing values
- Selected relevant Columns
-Fill the missing values

✔ Text Preprocessing

- Converted text to lowercase
- Removed URLs
- Removed special characters
- Removed stopwords
- Applied lemmatization

✔ Exploratory Data Analysis (EDA)

- Analyzed data distribution
- Identified patterns in complaints

✔ Visualization

- Generated WordCloud for text insights
- Created plots using Matplotlib & Seaborn

✔ NLP Feature Extraction

- Unigram analysis
- Bigram analysis
- CountVectorizer implementation


🛠 Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK


📈 Key Observations

- Large volume of textual complaints requires preprocessing
- Frequent terms identified using n-grams
- WordCloud helps visualize dominant complaint topics

👥 Team Contributions

- Data Collection, Cleaning,Text Preprocessing(Remove URLs,lowercase,Stopwords and special characters & applied Lemmatization) and EDA : Self
-Additional EDA & Word Clouds, n-grams (Unigram,bigram): Teammate
- Remaining tasks: In progress

📁 Project Structure

project/
│── week1 government data.ipynb
│── README.md
│── .gitignore

