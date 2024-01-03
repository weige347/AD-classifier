import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

# Function to print evaluation metrics and incorrect predictions
def print_metrics_with_names(y_true, y_pred, model_name, sample_names):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)  # Calculate AUC
    print(f'{model_name} Metrics:')
    print('Accuracy:', acc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)
    print('AUC:', auc)

    return acc, precision, recall, f1, auc


df_test = pd.read_csv('../test.csv')

label_dict = {'AD': 0, 'NC': 1}

y_test = df_test.iloc[0, 1:].values
y_test = np.vectorize(label_dict.get)(y_test)
X_test = df_test.iloc[1:, 1:].values
X_test = X_test.transpose()

models = {
    'Random Forest': RandomForestClassifier()
}
# Directory to save models
model_dir = '../weights'

# Create a DataFrame to store results
results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'])

# Plot all confusion matrices in a single figure
fig, axes = plt.subplots(2, 4, figsize=(15, 10))
fig.suptitle('Confusion Matrices for Different Models')

# Load models and plot confusion matrices
for (model_name, model), ax in zip(models.items(), axes.flatten()):
    model_filename = os.path.join(model_dir, f'{model_name}_model.joblib')
    loaded_model = joblib.load(model_filename)

    # Calculate predicted probabilities for positive class
    y_pred_proba = loaded_model.predict_proba(X_test)[:, 1]

    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Get evaluation metrics and incorrect predictions
    acc, precision, recall, f1, auc = print_metrics_with_names(y_test, y_pred, model_name, df_test.iloc[95, 1:].values)

    # Append results to DataFrame
    results_df = results_df.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc
    }, ignore_index=True)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(ax, cm, labels=['AD', 'NC'], title=model_name)

# Save results to Excel
results_df.to_excel('../results.xlsx', index=False)

