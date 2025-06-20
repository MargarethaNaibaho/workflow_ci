import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import argparse
import os

# 🧠 Hanya set tracking URI kalau belum di-set oleh MLflow run
if mlflow.get_tracking_uri() == "file:///default/artifact/root":
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

# Ambil argumen
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# Load data
df = pd.read_csv(args.data_path)
X = df.drop('income', axis=1)
y = df['income']

# Split dan normalisasi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Setup grid search
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

# Fungsi utama untuk training dan log
def run_training():
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(best_model, "model")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Jalankan dalam MLflow run
if mlflow.active_run() is None:
    with mlflow.start_run():
        run_training()
else:
    run_training()
