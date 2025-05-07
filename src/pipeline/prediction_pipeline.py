import sys
import pandas as pd
from src.components.exception import CustomException
from src.components.logger import logging
from src.components.utils import Load_Object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
          model_path = "src/components/artifact/model.pkl"
          preprocessor_path = "src/components/artifact/preprocessor.pkl"
          model=Load_Object(filepath=model_path)
          preprocessor=Load_Object(filepath=preprocessor_path)
          data_scaled=preprocessor.transform(features)
          pred=model.predict(data_scaled)
          return pred
        except Exception as e:
            raise CustomException(e,sys)
             


class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, test_preparation_course: str, lunch: str, writing_score: int, reading_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.test_preparation_course = test_preparation_course
        self.lunch = lunch
        self.writing_score = writing_score
        self.reading_score = reading_score
    def convert_to_df(self):
        try:
            dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "test preparation course": [self.test_preparation_course],
                "lunch": [self.lunch],
                "writing score": [self.writing_score],
                "reading score": [self.reading_score]
            }
            return pd.DataFrame(dict)
        except Exception as e:
            raise CustomException(e, sys)   
        


