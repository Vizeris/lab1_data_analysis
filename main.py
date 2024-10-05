import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import time
import sys

def log_message(message, log_file):
    with open(log_file, 'a') as f:
        f.write(message + '\n')
    print(message)

def load_and_prepare_data(file_path, target_column, log_file):
    data = pd.read_csv(file_path)
    log_message(f"Columns: {data.columns}", log_file)
    log_message(f"Dataset size: {data.shape}", log_file)

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    if data.isnull().sum().any():
        data = data.dropna()

    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def visualize_data(X, y, output_dir):
    plt.figure(figsize=(12, 8))
    corr = X.corr().clip(lower=0, upper=1)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title('Feature Correlation Matrix')
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.show()

    num_features = len(X.columns)
    fig, axs = plt.subplots(num_features, 2, figsize=(12, num_features * 5))

    for i, column in enumerate(X.columns):
        sns.histplot(X[column], kde=True, ax=axs[i, 0])
        axs[i, 0].set_title(f'Histogram for {column}')

        sns.boxplot(x=y, y=X[column], ax=axs[i, 1])
        axs[i, 1].set_title(f'Boxplot for {column}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_histograms_and_boxplots.png"))
    plt.show()

def normalize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def plot_combined_confusion_matrices(y_test, predictions, model_names, output_dir):
    fig, axs = plt.subplots(1, len(predictions), figsize=(18, 6))
    for i, (y_pred, model_name) in enumerate(zip(predictions, model_names)):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[i])
        axs[i].set_title(f'{model_name} Confusion Matrix')
        axs[i].set_xlabel('Predicted')
        axs[i].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_confusion_matrix.png"))
    plt.show()

def train_and_evaluate_models(X_train, X_test, y_train, y_test, output_dir, log_file):
    models = {
        "kNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42)
    }

    predictions = []
    model_names = []
    all_classification_reports = ""

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
        model_names.append(model_name)

        accuracy = accuracy_score(y_test, y_pred)
        log_message(f'Accuracy ({model_name}): {accuracy:.2f}', log_file)

        report = classification_report(y_test, y_pred)
        all_classification_reports += f"\nClassification Report ({model_name}):\n{report}\n"
        log_message(f"Classification Report ({model_name}):\n{report}", log_file)

    plot_combined_confusion_matrices(y_test, predictions, model_names, output_dir)

def optimize_knn(X_train, y_train, output_dir, log_file):
    param_grid = {'n_neighbors': np.arange(1, 31)}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)

    log_message(f"Best parameter for kNN: {grid.best_params_}", log_file)

    mean_test_scores = grid.cv_results_['mean_test_score']
    plt.figure(figsize=(10, 6))
    plt.plot(param_grid['n_neighbors'], mean_test_scores, marker='o')
    plt.title('kNN Accuracy vs. Number of Neighbors')
    plt.xlabel('Number of Neighbors (n_neighbors)')
    plt.ylabel('Mean Accuracy (Cross-Validation)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "knn_accuracy_vs_neighbors.png"))
    plt.show()

    return grid.best_estimator_

def optimize_svm(X_train, y_train, output_dir, log_file):
    param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)

    start_time = time.time()

    original_stdout = sys.stdout
    with open(log_file, 'a') as f:
        sys.stdout = f
        grid.fit(X_train, y_train)
        sys.stdout = original_stdout

    end_time = time.time()

    total_time = end_time - start_time
    log_message(f"Total time for GridSearchCV: {total_time:.4f} seconds", log_file)
    log_message(f"Best parameters for SVM: {grid.best_params_}", log_file)
    return grid.best_estimator_

def analyze_data(file_path, target_column):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = f"./{dataset_name}_output"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "output.log")
    X, y = load_and_prepare_data(file_path, target_column, log_file)
    visualize_data(X, y, output_dir)
    X_scaled = normalize_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    knn_best = optimize_knn(X_train, y_train, output_dir, log_file)
    svm_best = optimize_svm(X_train, y_train, output_dir, log_file)
    train_and_evaluate_models(X_train, X_test, y_train, y_test, output_dir, log_file)


file_path = 'proc_heart_cleve_3_withheader.csv'
target_column = 'Disease'

try:
    analyze_data(file_path, target_column)
except Exception as e:
    print(e)

#https://www.kaggle.com/datasets/muhammetvarl/heart-disease-dataset (proc_heart_cleve_3_withheader\Disease)
#https://www.kaggle.com/datasets/simaanjali/phising-detection-dataset (Phising_Detection_Dataset\Phising)

