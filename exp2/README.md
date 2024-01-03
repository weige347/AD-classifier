

## 1. Installation

Clone this repo.

This code requires 

 `pip install pandas scikit-learn seaborn matplotlib numpy joblib`

## 2. Structure

- data/
  - data.xlsx
  - test.xlsx
- weights/
  - 8 models weights
- README.md
- train.py
- test.py

## 3. Data

In the data folder, we provide two tables, where the data table is all the data information, and to ensure that our results are reproducible, we additionally save our test set in the test table



## 4. Inference Using Pretrained Model

Run the `test.py` file to test the model and generate evaluation metrics 

`python test.py`



## 5 . Train Model

Run the `train.py` file to train the model:

`python train.py`



