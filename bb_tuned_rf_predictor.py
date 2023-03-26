"""
Real Predictor for Unknown Test Data: MyBlueBerryNights Hackamecum Competition
Check with Test Set provided with Known (Labelled) Date

You can run this example as follows:
    $ python <filename>.py
"""

""" For Test DataSet Preparation """
from bb_trainer_handler import bb_dataset
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # El filtrado de NAs hace la serie excesivamente  

import pickle
import yaml
import sklearn.ensemble

def best_estimator_loader():
    with open("./best_model.pkl","rb") as fmodel:
        return pickle.load(fmodel)

def best_estimator_predictions(features):
    best_estimator = best_estimator_loader()
    with open("./best_model_features.yaml","r") as f:
        specs = yaml.load(f)
    th_proba = specs["ROC"]
    predictions = best_estimator.predict_proba(features)[:,1]
    return np.where(predictions>th_proba,1,0)

if __name__ == "__main__":
    # Data Preparation Block (class bb_dataset includes manipulations needed)
    features_reduced_set = ["Emayor", "Exc", "Rat"]
    features_dropped = ["Area","Vol","Perim","Emenor"]
    data_dict = {} # For bb_dataset initialization (dropping and imputing)
    
    impute_median = SimpleImputer(missing_values=np.nan, strategy="median")

    ct = ColumnTransformer(
        [   
            ("discards", "drop", features_dropped),
            ("imputer", impute_median, features_reduced_set)  
        ]
    )

    # Substitute with Real Test Competition Data
    # data = pd.read_excel("./datamecum_data/entrenamiento.xlsx")
    data = pd.DataFrame()

    data_dict["data"] = data

    # Unknown a priori on Test Competition Set
    # data_dict["target"] = data.iloc[:,-1].map({"A": 0, "B": 1})

    