from dataclasses import dataclass
import pandas as pd
import numpy as np
import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.components.exception import CustomException
from src.components.logger import logging
from src.components.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath=os.path.join("artifact","preprocessor.pkl")

class Data_Transformation:
    def __init__(self):
        self.Data_Transformation_Config=DataTransformationConfig()
    def get_data_transformer_object(self):

       # used to build the setup for transformation
        try:
            numerical_features=['writing score','reading score']
            categorical_features=['gender','race/ethnicity','test preparation course','parental level of education','lunch']

            num_pipeline=Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            cat_pipeline=Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder()),
                ('standardscalar',StandardScaler(with_mean=False))
            ])

            logging.info("Transformation of numerical and categorical features made into pipline")
            preprocessor=ColumnTransformer([
                ('num',num_pipeline,numerical_features),
                ('cat',cat_pipeline,categorical_features)
            ])

            return preprocessor
       

        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read the data ")
            logging.info("getting the preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column="math score"
            input_train_df= train_df.drop("math score",axis=1)
            target_train_df=train_df["math score"]

            input_test_df=test_df.drop(columns=[target_column],axis=1)
            target_test_df=test_df[target_column]
            logging.info("Applying the preprocesing")

            input_arr_train=preprocessing_obj.fit_transform(input_train_df)
            input_arr_test=preprocessing_obj.transform(input_test_df)

            train_arr=np.c_[input_arr_train,target_train_df]
            test_arr=np.c_[input_arr_test,target_test_df]
            logging.info("Transformation of data made ")
            
               
            save_object(
            filepath=self.Data_Transformation_Config.preprocessor_obj_filepath,  # Match attribute name
           obj=preprocessing_obj)
            
            return (train_arr,
                    test_arr,
                    self.Data_Transformation_Config.preprocessor_obj_filepath)

        except Exception as e1:
            raise CustomException(e1,sys)
            

    

    

