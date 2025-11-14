import numpy as np
import pandas as pd
import os
import sys
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)  #create directory if not present.

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)  #save the object to the file.
    except Exception as e:
        raise CustomException(e,sys)  #raise custom exception.


def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]

            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)


            # model.fit(X_train,y_train)  #fit the model.

            y_train_pred = model.predict(X_train)  #predict the train data.

            y_test_pred = model.predict(X_test)  #predict the test data.

            train_model_score = r2_score(y_train,y_train_pred)  #calculate r2 score.

            test_model_score = r2_score(y_test,y_test_pred)  #calculate r2 score.

            report[list(models.keys())[i]] = test_model_score  #store the model name and its score in the report dictionary.

        return report
    
    except Exception as e:
        raise CustomException(e,sys)  #raise custom exception.  