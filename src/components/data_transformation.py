import numpy as np
import pandas as pd
import sys
import os

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer #used to apply different transformations to different columns of the data.
from sklearn.impute import SimpleImputer  #used to fill missing values in the data.
from sklearn.pipeline import Pipeline  #used to create a pipeline of transformations.
from sklearn.preprocessing import OneHotEncoder,StandardScaler  #used for categorical and numerical data preprocessing respectively.
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass   #data class is a decorator that is used to create data classes in Python.
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')  #path to save the preprocessor object.

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() #initializing the data transformation config class.

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            logging.info('Data Transformation initiated')
            #numerical columns
            numerical_columns = ['reading_score','writing_score']

            #categorical columns
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',  
                'lunch',
                'test_preparation_course'
                ]
            
            #numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')), #filling missing values with median.
                    ('scaler',StandardScaler()) #scaling the numerical values.
                ]
            )

            #categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')), #filling missing values with most frequent values.
                    ('one_hot_encoder', OneHotEncoder()),  #one hot encoding the categorical values.
                    ('scaler', StandardScaler(with_mean=False))  #scaling the categorical values.
                ]
            )

            logging.info(f'Numerical Columns:  {numerical_columns}')
            logging.info(f'Categorical columns: {categorical_columns}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
    
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)  #reading train data
            test_df = pd.read_csv(test_path)   #reading test data

            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessor object')

            pre_processing_obj = self.get_data_transformer_object()  #getting the preprocessor object.

            target_column_name = 'math_score'
            numerical_columns = ['reading_score','writing_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)  #dropping target column from train data.
            target_feature_train_df = train_df[target_column_name]  #target column for train data.

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)  #dropping target column from test data.
            target_feature_test_df = test_df[target_column_name]  #target column for test data.

            logging.info('Applying preprocessing object on training and testing dataframes.')

            input_feature_train_arr = pre_processing_obj.fit_transform(input_feature_train_df)  #fitting and transforming the train data.
            input_feature_test_arr = pre_processing_obj.transform(input_feature_test_df)  #transforming the test data.

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
              ]  #combining input and target feature for train data.
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
              ]  #combining input and target feature for test data.
            
            logging.info('Saved preprocessing object.') 

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = pre_processing_obj
            )  #saving the preprocessor object.

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path  #returning train array, test array and preprocessor object file path.
            )
        except Exception as e:
            raise CustomException(e,sys)



        
