import os
import sys
from dataclasses import dataclass
from src.components.exception import CustomException
from src.components.logger import logging
from src.components.utils import save_object
from src.components.utils import evaluate_model

from sklearn .linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor 
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


@dataclass
class ModelTrainingConfig:
  trained_model_filepath=os.path.join("artifact","model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_training_config=ModelTrainingConfig()

  def  initiate_model_training(self,train_array,test_array):
    try:
      logging.info("Splitting training and testing data")

      X_train,y_train,X_test,y_test=(
        train_array[:,:-1],
        train_array[:,-1],
        test_array[:,:-1],
        test_array[:,-1]      

      )
      logging.info("Training the model")

      models={"LinearRegression":LinearRegression(),
        "Ridge":Ridge(),
        "Lasso":Lasso(),
        "KNeighborsRegressor":KNeighborsRegressor(),
        "DecisionTreeRegressor":DecisionTreeRegressor(),
        "RandomForestRegressor":RandomForestRegressor(),
        "GradientBoostingRegressor":GradientBoostingRegressor(),
        "AdaBoostRegressor":AdaBoostRegressor(),
        "XGBRegressor":XGBRegressor(),
        "CatBoostRegressor":CatBoostRegressor()
        }
      model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

      best_model_score=max(sorted(list(model_report.values())))

      best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

      best_model=models[best_model_name]

      if(best_model_score<0.7):
        raise CustomException("No best model found")
      logging.info("Training done  and best model found")

      save_object(
        filepath=self.model_training_config.trained_model_filepath,
        obj=best_model
      )

      predicted=best_model.predict(X_test)
      score=r2_score(y_test,predicted)
      print(f"The best model found is {best_model_name} and the r2 score is {score}")
      
      return score
     

      



    except Exception as e:
      raise CustomException(e,sys)
      


  





