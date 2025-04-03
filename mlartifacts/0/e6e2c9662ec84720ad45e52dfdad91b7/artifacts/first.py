import mlflow # To track experiments
import mlflow.sklearn # To automatically log models created using scikit-learn
from sklearn.datasets import load_wine # To load the wine dataset
from sklearn.ensemble import RandomForestClassifier # To create a random forest classifier
from sklearn.model_selection import train_test_split # To split the data into training and testing sets
from sklearn.metrics import accuracy_score, confusion_matrix # To evaluate the model's performance and create a confusion matrix to visualize the results
import matplotlib.pyplot as plt # To plot the confusion matrix
import seaborn as sns # To add color and style to the confusion matrix plot

mlflow.set_tracking_uri('http://localhost:5000')

# Load the wine dataset
wine = load_wine() 
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the parameters for the random forest classifier model
max_depth = 5
n_estimators = 10

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

    # mlflow.set_tags({})