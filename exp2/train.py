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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder

# Function to plot confusion matrix
def plot_confusion_matrix(ax, cm, labels, title):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

# Function to print evaluation metrics
def print_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'{model_name} Metrics:')
    print('Accuracy:', acc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)
    print('\n')

data = pd.read_excel("../data.xlsx")

# 选择特征和目标列
features = data.iloc[1:, 3:]  # 选择从第二行开始的所有行，从第四列开始的所有列
target = data['分组'][1:]  # 选择从第二行开始的目标列

# 将类别信息编码为0和1
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# 处理性别列
gender_column = pd.get_dummies(features['性别'], prefix='Gender', drop_first=True)
features = pd.concat([features, gender_column], axis=1)
features.drop('性别', axis=1, inplace=True)

# 读取数据
X_train, X_test, y_train, y_test, sample_names_train, sample_names_test = train_test_split(
    features, target_encoded, data['样品名称'][1:], test_size=0.2, random_state=42
)

# Define models
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(136,), max_iter=1000, alpha=0.001),
    'Naive Bayes': GaussianNB()
}

# Directory to save models and results
model_dir = '../saved_models_modify'
results_dir = '../model_results'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


# Evaluate models, save models, and print metrics
for model_name, model in models.items():
    model.fit(X_train, y_train)

    # Save the trained model
    model_filename = os.path.join(model_dir, f'{model_name}_model.joblib')
    joblib.dump(model, model_filename)

    # Load the model for testing
    loaded_model = joblib.load(model_filename)

    # Test the loaded model
    y_pred = loaded_model.predict(X_test)

    # Calculate and print metrics
    print_metrics(y_test, y_pred, model_name)


