import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from titanic_model import __version__ as _version
from titanic_model.config.core import config
from titanic_model.pipeline import titanic_pipe
from titanic_model.processing.data_manager import load_pipeline
from titanic_model.processing.data_manager import pre_pipeline_preparation


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
titanic_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data: dict) -> dict:
    """Make a prediction using a saved model """

    data = pre_pipeline_preparation(data_frame=pd.DataFrame(input_data))
    data=data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    
    results = {"predictions": None, "version": _version, }
    
    predictions = titanic_pipe.predict(data)

    results = {"predictions": predictions,"version": _version}
    #print(results, results["predictions"][0])

    return results

if __name__ == "__main__":

    data_in={'PassengerId':[79],'Pclass':[2],'Name':["Caldwell, Master. Alden Gates"],'Sex':['male'],'Age':[0.83],
                'SibSp':[0],'Parch':[2],'Ticket':['248738'],'Cabin':[np.nan,],'Embarked':['S'],'Fare':[29]}
    
    make_prediction(input_data=data_in)