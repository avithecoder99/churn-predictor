import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Sample dummy dataset
data = {
    "age": [25, 45, 35, 33, 60],
    "balance": [1000, 2000, 3000, 4000, 5000],
    "churn": [0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# Split
X = df[["age", "balance"]]
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
os.makedirs("app/model", exist_ok=True)
joblib.dump(model, "app/model/churn_model.pkl")
