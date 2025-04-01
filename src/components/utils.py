import pandas as pd
import numpy as np
import os
from src.components.exception import CustomException
import dill
 
def save_object(filepath,obj):
    try:
        dir_name =os.path.dirname(filepath)
        os.makedirs(dir_name,exist_ok=True)

        with open(filepath,'wb') as file_obj:
            dill.dump(obj,file_obj)
        
    except:
        pass
 
    
    
    
      

    
        


