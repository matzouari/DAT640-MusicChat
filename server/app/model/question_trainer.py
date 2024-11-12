import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset
data = pd.read_csv("question_data.csv")

# Split into features (questions) and labels (intents)
X = data["question"]
y = data["intent"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and SVM classifier
pipeline = make_pipeline(TfidfVectorizer(), SVC(kernel="linear", probability=True))

# Train the SVM model
pipeline.fit(X_train, y_train)

# Evaluate the model
accuracy = pipeline.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(pipeline, "question_classifier_model.pkl")
