import os,sys
from src.components.logger import logging
from src.components.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import Data_Transformation


 # stores the information like where diffrent type of data is stored
@dataclass
class DataIngestionConfig: 
    train_data_path:str=os.path.join("artifact","train.csv")
    test_data_path:str=os.path.join("artifact","test.csv")
    raw_data_path:str=os.path.join("artifact","raw.csv")



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingested method")
        try :
            df=pd.read_csv(r"C:\Users\DELL\Desktop\ML project\StudentsPerformance.csv")
            logging.info("read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #create a folder if the path does'nt exist for train
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
        
            logging.info("starting train test split")

            train_df,test_df=train_test_split(df,test_size=0.2,random_state=42)

            train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion done")

            return (
                self.ingestion_config.train_data_path

                ,self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            
if __name__=="__main__":
    obj=DataIngestion()
    
    train_df,test_df=obj.initiate_data_ingestion()

    data_transformation=Data_Transformation()
    data_transformation.initiate_data_transformation(train_df,test_df) 

   
            



                        