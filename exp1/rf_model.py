import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os


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

df_train = pd.read_csv('/train.csv')
df_test = pd.read_csv('/test.csv')

# Train dataset
y_train = df_train.iloc[0, 1:].values
label_dict = {'AD': 0, 'NC': 1}
y_train = np.vectorize(label_dict.get)(y_train)
X_train = df_train.iloc[1:, 1:].values
X_train = X_train.transpose()

# Test dataset
y_test = df_test.iloc[0, 1:].values
y_test = np.vectorize(label_dict.get)(y_test)
X_test = df_test.iloc[1:, 1:].values
X_test = X_test.transpose()

# Define models
models = {
    'Random Forest': RandomForestClassifier(),
}

# Directory to save models
model_dir = 'saved_models_modify'
os.makedirs(model_dir, exist_ok=True)

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

# Plot all confusion matrices in a single figure
fig, axes = plt.subplots(2, 4, figsize=(15, 10))
fig.suptitle('Confusion Matrices for Different Models')

# Load models and plot confusion matrices
for (model_name, model), ax in zip(models.items(), axes.flatten()):
    model_filename = os.path.join(model_dir, f'{model_name}_model.joblib')
    loaded_model = joblib.load(model_filename)
    y_pred = loaded_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(ax, cm, labels=['AD', 'NC'], title=model_name)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
