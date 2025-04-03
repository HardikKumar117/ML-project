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
      
      params = {
    "LinearRegression": {
        "fit_intercept": [True, False],
        #"copy_X": [True, False],
        "n_jobs": [-1, None],
        "positive": [True, False]
    },
    "Ridge": {
        "alpha": [0.01, 0.1, 1, 10, 100],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "lbfgs"]
    },
    "Lasso": {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
        "max_iter": [1000, 5000, 10000]
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 10, 15],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "p": [1, 2]  # 1: Manhattan, 2: Euclidean
    },
    "DecisionTreeRegressor": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10]
    },
    "RandomForestRegressor": {
        "n_estimators": [10, 50, 100, 200, 500],
        "criterion": ["squared_error", "absolute_error"],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10],
        "bootstrap": [True, False]
    },
    "GradientBoostingRegressor": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.001, 0.01, 0.1, 0.2],
        "max_depth": [3, 5, 10, 20],
        "subsample": [0.5, 0.7, 1.0]
    },
    "AdaBoostRegressor": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.001, 0.01, 0.1, 1.0]
    },
    "XGBRegressor": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.1, 0.2, 0.3],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.5, 0.7, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0]
    },
    "CatBoostRegressor": {
        "iterations": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 10]
    }
}


      model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

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
      


  





