import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import mlflow

# 1. Load data from Hugging Face
try:
    dataset = load_dataset("sathyam123/tourismpackagenew")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_df = train_dataset.to_pandas()
    test_df = test_dataset.to_pandas()
    print("Data loaded successfully from Hugging Face.")
except Exception as e:
    print(f"Error loading data from Hugging Face: {e}")
    exit() # Exit if data loading fails

# 2. Prepare data for modeling
X_train = train_df.drop(columns=['ProdTaken', '__index_level_0__'])
y_train = train_df['ProdTaken']
X_test = test_df.drop(columns=['ProdTaken', '__index_level_0__'])
y_test = test_df['ProdTaken']

# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Apply one-hot encoding
X_train = pd.get_dummies(X_train, columns=categorical_cols)
X_test = pd.get_dummies(X_test, columns=categorical_cols)

# Align columns - crucial for consistent features
train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0

X_test = X_test[train_cols]

print("Data preprocessing complete.")
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# 3. Define models and parameters
models = {
    'Decision Tree': (DecisionTreeClassifier(random_state=42), {
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }),
    'Bagging': (BaggingClassifier(random_state=42), {
        'n_estimators': [50, 100],
        'max_samples': [0.8, 1.0],
        'max_features': [0.8, 1.0]
    }),
    'Random Forest': (RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }),
    'AdaBoost': (AdaBoostClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 1.0]
    }),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5]
    }),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    })
}

# Reduce parameter grids for faster execution in CI
for model_name in models:
    model_info = models[model_name]
    if isinstance(model_info[1], dict):
        for param in model_info[1]:
            if isinstance(model_info[1][param], list) and len(model_info[1][param]) > 2:
                models[model_name][1][param] = models[model_name][1][param][:2]


print("Models and simplified parameter grids defined.")

# 4. Tune and train models with MLflow logging
experiment_results = {}
mlflow.set_experiment("Tourism Package Prediction")

for model_name, (model, param_grid) in models.items():
    with mlflow.start_run(run_name=f"{model_name}_tuning"):
        print(f"\nTuning {model_name}...")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1) # Use F1 for tuning
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_

        # Log best parameters with MLflow
        mlflow.log_params(best_params)

        experiment_results[model_name] = {
            'best_params': best_params,
            'best_estimator': best_estimator
        }
        print(f"Finished tuning {model_name}. Best params: {best_params}")

print("\nHyperparameter tuning complete.")

# 5. Evaluate models with MLflow logging
print("\nEvaluating models...")
for model_name, results in experiment_results.items():
    with mlflow.start_run(run_name=f"{model_name}_evaluation"):
        model = results['best_estimator']

        # Make predictions on the testing data
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] # Get probabilities for AUC

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Log evaluation metrics with MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)


        # Store the evaluation metrics
        experiment_results[model_name]['evaluation_metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }

        # Print the evaluation metrics
        print(f"  {model_name} Metrics:")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1-score: {f1:.4f}")
        print(f"    AUC: {auc:.4f}")

print("\nModel evaluation complete.")

# 6. Identify the best performing model based on F1-score
best_model_name = None
best_f1_score = -1

print("\nDetermining the best model...")
for model_name, results in experiment_results.items():
    f1 = results['evaluation_metrics']['f1_score']
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model_name = model_name

print(f"Best performing model based on F1-score: {best_model_name} with F1-score: {best_f1_score:.4f}")

# 7. Save the best model locally
best_model = experiment_results[best_model_name]['best_estimator']
local_model_dir = "./best_model"
os.makedirs(local_model_dir, exist_ok=True)
model_path = os.path.join(local_model_dir, "model.joblib")
joblib.dump(best_model, model_path)
print(f"Best model saved locally to {model_path}")

# The workflow will handle pushing to Hugging Face Hub using upload-hub-action
print("\nML pipeline script finished. Model saved locally for upload.")
