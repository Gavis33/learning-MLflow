import mlflow
import mlflow.sklearn 
from sklearn.datasets import load_wine 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns

import dagshub
dagshub.init(repo_owner='Gavis33', repo_name='learning-MLflow', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/Gavis33/learning-MLflow.mlflow')

# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the parameters for the random forest classifier model
max_depth = 10
n_estimators = 10

mlflow.set_experiment('MLflow_exp1_using_dagshub')

with mlflow.start_run(): # context manager to track experiments
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_params({"max_depth": max_depth, "n_estimators": n_estimators})

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    plt.savefig('confusion_matrix.png')

    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)

    mlflow.set_tags({'Author': 'Gavis', 'Project': 'Wine Classifier'})
    
    mlflow.sklearn.log_model(rf, "RandomForestClassifier")