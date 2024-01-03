

## 1. Installation

Clone this repo.

This code requires 

 `pip install pandas scikit-learn seaborn matplotlib numpy joblib`

## 2. Structure

- data/
  - train.csv
  - test.csv
- weights/
  - Random Forest_model.joblib
- README.md
- train.py
- test.py

## 3. Data

 In the data folder, we provide the 97 think features we used on the training as well as the test dataset.20 cases in train and 13 cases in test.



## 4. Inference Using Pretrained Model

Run the `test.py` file to test the model and generate evaluation metrics and confusion matrices:

`python test.py`



## 5 . Train Model

Run the `train.py` file to train the model:

`python train.py`



