import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
                            RandomForestClassifier, 
                            AdaBoostClassifier,
                            GradientBoostingClassifier
                            )

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, save_object, evaluate_model 


# @dataclass
# class ModelTrainerConfig:
#     model_obj_file_path=os.path.join('artifacts',"model.pkl")

# class ModelTrainer: 
#     def __init__(self, config: ModelTrainerConfig):
#         self.config = config

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    
    def __init__(self, config: ModelTrainerConfig):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
         
         try:
            logging.info("Splitting dependent and independent variables from train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Successfully dependent and independent splitted in training and testset")

            logging.info("Ready to do model")

            models = {
                "Logistic Regression" : LogisticRegression(),
                "Random Forest": RandomForestClassifier(n_jobs=-1),
    
            }

            logging.info("Successfully Model was fitted")

            logging.info("Evaluation Model")

            model_report, train_model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            logging.info(f'Train Model Report : {train_model_report}')

            ## To get the best model score from dictionary
            # best_model_score = max(sorted(model_report.values()))
            best_model_score = max(model_report.items(), key=lambda x: x[1]['model_test_f1'])
            
            best_model_name = max(model_report.items(), key=lambda x: x[1]['model_test_f1'])[0]
 
            # best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , F1_Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , F1_Score : {best_model_score}')

            




            save_object(
                    obj=best_model,
                    file_path=self.config.trained_model_file_path             
            )

         except Exception as e:
            logging.info("Exception occured at model training")
            raise CustomException(e, sys)

# @dataclass
# class ModelTrainerConfig:
#     model_obj_file_path=os.path.join('artifacts',"model.pkl")

# class ModelTrainer: 
#     def __init__(self, config: ModelTrainerConfig):
#         self.config = config 

#     def fit(self, train_array, test_array):
#         try : 
#             logging.info("Split training and test input data into X_train, X_test, y_train, y_test")
#             X_train,y_train,X_test,y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1]) 
#             models = {
#                 "LogisticRegression": LogisticRegression(),
#                 "RandomForestClassifier": RandomForestClassifier(),
#                 "AdaBoostClassifier": AdaBoostClassifier(),
#                 "GradientBoostingClassifier": GradientBoostingClassifier()
#             }

#             model_params = { "LogisticRegression": {'class_weight': ['balanced', None], 'penalty': ['l1', 'l2', 'elasticnet', 'none'], 
#                                                     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
#                                                     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
#                                                     'max_iter': [100, 1000, 2500, 5000]},
#                             "RandomForestClassifier": {'n_estimators': [100, 200, 300, 400, 500],
#                                                         'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, None],
#                                                         'min_samples_split': [2, 5, 10, 15, 100],
#                                                         'min_samples_leaf': [1, 2, 5, 10]},
#                                                         'criterion': ['gini'],
#                                                         'n_jobs': [-1],
#                                                         'bootstrap': [True],
#                                                         'ooB_score': [True],
#                                                         'class_weight': ['balanced', 'balanced_subsample', None],
#                             "AdaBoostClassifier": {'n_estimators': [50, 100, 150, 200, 250, 300],
#                                                     'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
#                                                     'algorithm': ['SAMME', 'SAMME.R']},
#                                                     'class_weight': ['balanced', None],
#                             "GradientBoostingClassifier": {'loss': ['deviance', 'exponential'],
#                                                                 'learning_rate': [0.1, 0.05, 0.01, 0.001],
#                                                                 'n_estimators': [100, 150, 200, 250, 300],
#                                                                 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#                                                                 'min_samples_split': [2, 5, 10, 15, 100],
#                                                                 'min_samples_leaf': [1, 2, 5, 10],
#                                                                 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 
#                                                                 'max_features': ['auto', 'sqrt', 'log2'],
#                                                                 'class_weight': ['balanced', 'balanced_subsample', None]
#                                                                 }
#                             }
#             logging.info("Getting the data") 
           
#             model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models, model_params) 
#             logging.info("Model report is generated")
#              ## To get best model score from dictionary
#             best_model_score = 0
#             for model_name in model_report:
#                 if model_report[model_name]['accuracy_score'] > best_model_score:
#                     best_model_score = model_report[model_name]['accuracy_score']
#                     best_model_name = model_name 

#             logging.info("Best model name is {}".format(best_model_name)) 
#             logging.info("Best model score is {}".format(best_model_score))

#             # Get the best model name 
#             self.config.model_name = best_model_name 

#             # Get the best model params 
#             self.config.model_params = model_params[best_model_name] 

#             # Fit the model on the training data with best params

#             self.model = models[best_model_name].set_params(**model_params[best_model_name]) 
#             self.model.fit(X_train, y_train) 
#             logging.info("Model is fitted on the training data")

#             # Fit the model on the test data 

#             y_train_pred = self.model.predict(X_train)
#             y_test_pred = self.model.predict(X_test) 

#             logging.info("Model is fitted on the test data")

#             # Evaluate the model on the test data

#             report = {
#                 "accuracy_score": accuracy_score(y_test, y_test_pred),
#                 "precision_score": precision_score(y_test, y_test_pred),
#                 "recall_score": recall_score(y_test, y_test_pred),
#                 "f1_score": f1_score(y_test, y_test_pred),
#                 "confusion_matrix": confusion_matrix(y_test, y_test_pred),
#                 "classification_report": classification_report(y_test, y_test_pred)
#             } 

#             logging.info("Model is evaluated on the test data")

#             # Save the model object

#             save_object(self.model, self.config.model_obj_file_path) 

#             logging.info("Model object is saved in the path: {}".format(self.config.model_obj_file_path)) 

#             return report 
        
#         except Exception as e:
#             raise CustomException(e, sys)





if __name__ == "__main__":
    data_transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation(config=data_transformation_config)
    train_array, test_array = data_transformation.transform_data('artifacts/train_data.csv', 'artifacts/test_data.csv', 'artifacts/raw_data.csv')
    logging.info("Data Transformation has been ended")
    model_trainer_config = ModelTrainerConfig()
    model_trainer = ModelTrainer(config=model_trainer_config)
    model_trainer.initiate_model_training(train_array, test_array)