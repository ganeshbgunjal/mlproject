import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    )

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass   #data class is a decorator that is used to create data classes in Python.
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')  #path to save the trained model.

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()  #initializing the model trainer config class.

    def initiate_model_trainer(self,train_aray, test_array, preprocessor_path):
        try:
            logging.info('Split training and test input data')
            X_train,y_train,X_test,y_test = (
                train_aray[:,:-1],
                train_aray[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor()
            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,
                            y_test=y_test,models=models)  #evaluating all the models.
            
            #to get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            #to get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")  #raising custom exception if no best model found. 
            
            logging.info(f'Best model found: {best_model_name} with r2 score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )  #saving the best model object.

            predicted = best_model.predict(X_test)  #predicting the test data.
            r2_square = r2_score(y_test,predicted)  #calculating r2 score.
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)
        
