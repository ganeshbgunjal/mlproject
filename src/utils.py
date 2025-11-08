import numpy as np
import pandas as pd
import os
import sys
import dill

from src.exception import CustomException
from src.logger import logging


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)  #create directory if not present.

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)  #save the object to the file.
    except Exception as e:
        raise CustomException(e,sys)  #raise custom exception.


if __name__ == '__main__':
    obj = DataIngestion()
    save_object('artifacts/test.pkl',obj)