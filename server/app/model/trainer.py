import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("./data.csv")

# Vectorize the text using CountVectorizer (ngrams up to 2)
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df['phrase'])
y = df['intent']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer for later use
import joblib
joblib.dump(clf, 'intent_classifier_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
