import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the training dataset
data = pd.read_csv('training_data.csv')

# Separate features and labels
X = data[[
    'num_columns', 'num_rows', 'num_numerical', 'num_categorical',
    'has_datetime', 'has_geo', 'avg_unique_values', 'correlation_score'
]]

y = data[['bar', 'pie', 'line', 'scatter', 'table', 'map']]

# Split dataset for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
model = MultiOutputClassifier(base_model)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Evaluation:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Save model
joblib.dump(model, 'chart_model.pkl')
print("Model saved as 'chart_model.pkl'")
