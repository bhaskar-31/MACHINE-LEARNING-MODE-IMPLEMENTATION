import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
# Dataset: SMS Spam Collection Dataset (can be downloaded from UCI ML repo or Kaggle)
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]

# Encode labels: ham=0, spam=1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 2. Preprocessing
X = df["message"]
y = df["label"]

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predictions
y_pred = model.predict(X_test)

# 5. Evaluation
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 6. Test on new email
new_message = ["Congratulations! You won a $1000 Walmart gift card. Call now!"]
new_message_vec = vectorizer.transform(new_message)
prediction = model.predict(new_message_vec)
print("Prediction:", "Spam" if prediction[0] == 1 else "Ham")
