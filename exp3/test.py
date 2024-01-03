import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib  # Added for model loading
import os
# Load models from the directory MOCA models
linear_model = joblib.load('../MOCA_models/linear_model.pkl')
tree_model = joblib.load('../MOCA_models/tree_model.pkl')
rf_model = joblib.load('../MOCA_models/rf_model.pkl')

file_path = 'D:\\mywork\\blood_AD\\excel_test2.xlsx'
data = pd.read_excel(file_path)
sample_names = data['样品名称']
selected_columns = ['性别', '年龄', 'Mg (ppm)', 'Ca (ppm)', 'Cr (ppb)', 'Mn (ppb)', 'Fe (ppm)',
                     'Co (ppb)', 'Ni (ppb)', 'Cu (ppm)', 'Zn (ppm)', 'As (ppb)', 'Se (ppb)', 'Sr (ppb)',
                     'Mo (ppb)', 'Cd (ppb)', 'Sn (ppb)', 'Sb (ppb)', 'I (ppb)',  'Hg (ppb)',
                     'Pb (ppb)', 'Bi (ppb)', 'Cu/Zn']
features = data[selected_columns]
labels = data['MOCA']
#labels = data['MMSE']
gender_column = pd.get_dummies(features['性别'], prefix='Gender', drop_first=True)
features = pd.concat([features, gender_column], axis=1)
features.drop('性别', axis=1, inplace=True)

X_train, X_test, y_train, y_test, sample_names_train, sample_names_test = train_test_split(features, labels, sample_names, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_linear = y_pred_linear.round().astype(int)

y_pred_tree = tree_model.predict(X_test_scaled)
y_pred_tree = y_pred_tree.round().astype(int)

y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_rf = y_pred_rf.round().astype(int)

wrong_predictions_linear = sample_names_test[y_test != y_pred_linear]
wrong_predictions_tree = sample_names_test[y_test != y_pred_tree]
wrong_predictions_rf = sample_names_test[y_test != y_pred_rf]

output_dir = '../result/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


result_df = pd.DataFrame({
    'Sample_Name': sample_names_test.tolist(),
    'Actual_Values': y_test.tolist(),
    'Linear_Regression_Predictions': y_pred_linear.tolist(),
    'Decision_Tree_Predictions': y_pred_tree.tolist(),
    'Random_Forest_Predictions': y_pred_rf.tolist(),
    'Linear_Regression_Difference': y_test - y_pred_linear,
    'Decision_Tree_Difference': y_test - y_pred_tree,
    'Random_Forest_Difference': y_test - y_pred_rf
})


excel_writer = pd.ExcelWriter(os.path.join(output_dir, 'result.xlsx'), engine='xlsxwriter', options={'strings_to_numbers': True, 'strings_to_formulas': False})
result_df.to_excel(excel_writer, sheet_name='result', index=False, encoding='utf-8')

excel_writer.save()