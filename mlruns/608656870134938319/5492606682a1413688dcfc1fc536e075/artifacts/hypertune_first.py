import mlflow
import mlflow.data
import mlflow.data
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV # For hyperparameter tuning 
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# estimator=rf means use the RandomForestClassifier as the base estimator
# cv=5 means use 5-fold cross-validation to evaluate the performance of the model in simple words: split the data into 5 equal parts and train the model on 4 parts and evaluate it on the remaining part
# n_jobs=-1 means use all available CPU cores
# verbose=2 means print out more information

# ----without mlflow ----
# grid_search.fit(X_train, y_train)

# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print("Best Parameters:", best_params)
# print("Best Score:", best_score)


# ----with mlflow ----

mlflow.set_experiment('Breast-cancer-hyperparameter-tuning')

with mlflow.start_run() as parent:

    grid_search.fit(X_train, y_train)

    # log all the child runs as a single run
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_score)

    # log train and test datasets
    train_df = X_test.copy()
    train_df['target'] = y_train
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, 'training')

    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, 'testing')
 
    # logging other stuff
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(grid_search.best_estimator_, 'random_forest')

    mlflow.set_tag('Author', 'Gavis')

    print(best_params)
    print(best_score)