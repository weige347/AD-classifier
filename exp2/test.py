import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder
def print_metrics(y_true, y_pred, y_prob, model_name):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_value = roc_auc_score(y_true, y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f'{model_name} Metrics:')
    print('Accuracy:', acc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)
    print('AUC:', auc_value)
    print('\n')
# Load data
data = pd.read_excel("../data.xlsx/")

# Select features and target column
features = data.iloc[1:, 3:]  # Select all rows from the second row and all columns from the fourth column
target = data['分组'][1:]  # Select the target column from the second row

# Encode categorical information into 0 and 1
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Process gender column
gender_column = pd.get_dummies(features['性别'], prefix='Gender', drop_first=True)
features = pd.concat([features, gender_column], axis=1)
features.drop('性别', axis=1, inplace=True)

# Read data
X_train, X_test, y_train, y_test, sample_names_train, sample_names_test = train_test_split(
    features, target_encoded, data['样品名称'][1:], test_size=0.2, random_state=42
)
# Define models
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(136,), max_iter=1000, alpha=0.001),
    'Naive Bayes': GaussianNB()
}

# Directory to save models, results, and ROC curves
model_dir = '../model_dir/'

# Load pre-trained models and evaluate on the test set
for model_name, model in models.items():
    # Load the pre-trained model
    model_path = os.path.join(model_dir, f'{model_name}_model.joblib')
    loaded_model = joblib.load(model_path)

    # Evaluate the model on the test set
    y_pred = loaded_model.predict(X_test)
    y_prob = loaded_model.predict_proba(X_test)[:, 1]

    # Print evaluation metrics
    print_metrics(y_test, y_pred, y_prob, model_name)