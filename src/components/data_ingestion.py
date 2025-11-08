# show me code from for the file src/components/data_ingestion.py
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass #used to create data classes in Python.

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass  #data class is a decorator that is used to create data classes in Python.
class DataIngestionConfig:  
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):  #read data from source and split into train and test.
        logging.info('entered the data ingestion method or components')

        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42) # splitting the data into train and test.

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)  #saving train data to artifacts folder.

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of the data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )  #returning train and test data path.



        except Exception as e:
            raise CustomException(e,sys) #raising custom exception.
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()  #calling the data ingestion method.

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)