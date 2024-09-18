import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        best_model = None  
        best_model_score = float('-inf')
        for model_name,model in models.items():
            para = param[model_name]

            # Perform Grid Search with Cross-Validation
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            # Get the best model from GridSearchCV
            model = gs.best_estimator_

            # Evaluate the model on training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            if test_model_score > best_model_score:
                best_model_score = test_model_score
                best_model = model
        return report,best_model,best_model_score

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)