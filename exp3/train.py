import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import joblib

file_path = '../data.xlsx'  
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

linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_linear = y_pred_linear.round().astype(int)

tree_model = DecisionTreeRegressor()
tree_model.fit(X_train_scaled, y_train)
y_pred_tree = tree_model.predict(X_test_scaled)
y_pred_tree = y_pred_tree.round().astype(int)

rf_model = RandomForestRegressor()
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_rf = y_pred_rf.round().astype(int)


joblib.dump(linear_model, 'D:\mywork/blood_AD\exp2\MOCA_models/linear_model.pkl')
joblib.dump(tree_model, 'D:\mywork/blood_AD\exp2\MOCA_models/tree_model.pkl')
joblib.dump(rf_model, 'D:\mywork/blood_AD\exp2\MOCA_models/rf_model.pkl')
