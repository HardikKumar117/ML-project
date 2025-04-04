import pandas as pd
import numpy as np
import os
import sys
from src.components.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import dill

 
def save_object(filepath,obj):
    try:
        dir_name =os.path.dirname(filepath)
        os.makedirs(dir_name,exist_ok=True)

        with open(filepath,'wb') as file_obj:
            dill.dump(obj,file_obj)        
    except Exception as e :
        raise CustomException(e,sys)

def evaluate_model(X_train,X_test,y_train,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]
            grid_search = GridSearchCV(model, para, cv=3)
            grid_search.fit(X_train,y_train)

            model.set_params(**grid_search.best_params_)


            model.fit(X_train,y_train)
            y_pred_train= model.predict(X_train)
            y_pred_test=model.predict(X_test)
            train_model_score=r2_score(y_train,y_pred_train)
            test_model_score=r2_score(y_test,y_pred_test)


            

            model_keys = list(models.keys())  # Convert dict_keys to a list
            report[model_keys[i]] = test_model_score  # Access the i-th key correctly




        return report  
    except Exception as e1:
        raise CustomException(e1,sys)
 
    
    
    
      

    
        


