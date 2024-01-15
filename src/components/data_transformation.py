import sys
import os
import logging
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.utils import save_object
from src.components.data_engineering import DataEngineeringConfig
from src.components.data_engineering import DataEngineering
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl") 

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

 
    def get_transformer_data(self): 
        try :
            numerical_features = ['loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade', 
                                  'emp_length', 'annual_inc', 'dti', 'open_acc', 
                                  'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 
                                  'mort_acc', 'pub_rec_bankruptcies'] 
            
            categorical_features = ['home_ownership', 'verification_status', 'purpose', 'application_type', 'zip_code']  

            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())]) 
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))]) 
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)]) 
            return preprocessor 
        
        except Exception as e:
            raise CustomException(e, sys) 
        

    def transform_data(self, train_data_path, test_data_path, raw_data_path): 
        # get the data from the source 
        try:
            # train_data = pd.read_csv(train_data_path)
            # test_data = pd.read_csv(test_data_path)
            # raw_data = pd.read_csv(raw_data_path)
            # logging.info("Data is read from the source")

            # get the preprocessor object
            logging.info("Creating the preprocessor object")
            preprocessor_obj = self.get_transformer_data() 
            logging.info("Preprocessor object is created")

            # fit the preprocessor object on the train data and first pass it from data_engineering 
            logging.info("Creating the data engineering object")
            data_engineering_config = DataEngineeringConfig()
            data_engineering = DataEngineering(config=data_engineering_config)
            logging.info("Data Engineering object is created and path is passed")
            train_data = data_engineering.preprocess_data(train_data_path)
            test_data = data_engineering.preprocess_data(train_data_path) 
            logging.info("Data is preprocessed") 

            # get the train and test data 
            train_df = train_data.drop(columns=['loan_status'], axis=1)
            test_df = test_data.drop(columns=['loan_status'], axis=1) 
            print(train_df.shape)
            print(test_df.shape)
            print(train_df.head())
            logging.info("Train and test data is created")
            target_train_df = train_data['loan_status'] 
            target_test_df = test_data['loan_status']
            # fit the preprocessor object on the train data
            train_df_array = preprocessor_obj.fit_transform(train_df)
            logging.info("Preprocessor object is fitted on the train data")
            test_df_array = preprocessor_obj.transform(test_df) 
            logging.info("Preprocessor object is fitted on the test data") 

            # logiging the shape of the train and test data
            logging.info("Shape of the train data: {}".format(train_df_array.shape))
            logging.info("Shape of the test data: {}".format(test_df_array.shape)) 

            # save the preprocessor object
            logging.info("Saving the preprocessor object")

            logging.info("logging the train_arr and test_arr by concatinating the target_train_df and target_test_df")

            train_arr = np.concatenate((train_df_array, target_train_df.values.reshape(-1,1)), axis=1) 
            test_arr = np.concatenate((test_df_array, target_test_df.values.reshape(-1,1)), axis=1)

            logging.info("Shape of the train_arr: {}".format(train_arr.shape))
            logging.info("train arr and test arr is created")

            save_object(preprocessor_obj, self.config.preprocessor_obj_file_path) 

            logging.info("Preprocessor object is saved in the path: {}".format(self.config.preprocessor_obj_file_path)) 

            return (train_arr, test_arr, self.config.preprocessor_obj_file_path) 
        
        except Exception as e:
            raise CustomException(e, sys)  
        


# if __name__ == "__main__": 
#     logging.info("Data Transformation has been started")
#     data_transformation_config = DataTransformationConfig()
#     data_transformation = DataTransformation(config=data_transformation_config)
#     data_transformation.transform_data('artifacts/train_data.csv', 'artifacts/test_data.csv', 'artifacts/raw_data.csv')
#     logging.info("Data Transformation has been ended")

        



