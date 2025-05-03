import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample dataset
data = {
    "description": [
        # AI Engineer job descriptions
        "Experience in deep learning and TensorFlow",
        "Proficient in natural language processing with spaCy",
        "Skilled in machine learning and data preprocessing",
        "Worked on computer vision models and OpenCV",
        "Developed models using PyTorch and neural networks",

        # Cloud Engineer job descriptions
        "Develop cloud solutions with AWS and Docker",
        "Deploy containerized applications using Kubernetes",
        "Manage infrastructure with Terraform and Azure",
        "Configure CI/CD pipelines for cloud deployment",
        "Maintain scalable systems on Google Cloud Platform"
    ],
    "role": [
        # Labels (matching descriptions above)
        "AI Engineer",
        "AI Engineer",
        "AI Engineer",
        "AI Engineer",
        "AI Engineer",
        "Cloud Engineer",
        "Cloud Engineer",
        "Cloud Engineer",
        "Cloud Engineer",
        "Cloud Engineer"
    ]
}

df = pd.DataFrame(data)

# Split
X_train, X_test, y_train, y_test = train_test_split(df["description"], df["role"], test_size=0.25, random_state=42)

# Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

