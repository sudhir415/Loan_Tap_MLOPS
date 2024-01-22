import os 
import sys

import pandas as pd
import numpy as np
from src.exception import CustomException
import dill  
import pickle 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from src.logger import logging

 
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_objt:
            return pickle.load(file_objt)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        report_train = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            ## Train Model
            model.fit(X_train, y_train)

            ## Predict Testing Data
            y_test_pred = model.predict(X_test)

            ## Get Accuracy, F1 Score, Precision & Recall for train and test data
            # train_model_score = r2_score(y_test, y_train_pred)

            model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
            model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score
            model_test_precision = precision_score(y_test, y_test_pred) # Calculate Precision
            model_test_recall = recall_score(y_test, y_test_pred) # Calculate Recall

            report[list(models.keys())[i]] = {"model_test_accuracy":model_test_accuracy, "model_test_f1":model_test_f1, 
                                              "model_test_precision":model_test_precision, "model_test_recall":model_test_recall}
            
            ## Predict Training Data
            y_train_pred = model.predict(X_train)

            ## Get Accuracy, F1 Score, Precision & Recall for train data and append in report_train dictionary
            train_model_score = accuracy_score(y_train, y_train_pred)
            train_model_f1 = f1_score(y_train, y_train_pred, average='weighted')
            train_model_precision = precision_score(y_train, y_train_pred)
            train_model_recall = recall_score(y_train, y_train_pred)

            report_train[list(models.keys())[i]] = {"train_model_score":train_model_score, "train_model_f1":train_model_f1, 
                                              "train_model_precision":train_model_precision, "train_model_recall":train_model_recall}

        return report, report_train 
    
    except Exception as e:
        logging.info("Exception occured during model training")
        raise CustomException(e, sys)
    
