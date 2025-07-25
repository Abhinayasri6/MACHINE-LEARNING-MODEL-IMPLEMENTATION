import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Using Logistic Regression as an example model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression(max_iter=200) # Increased max_iter for convergence
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("🔹 Accuracy Score:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
