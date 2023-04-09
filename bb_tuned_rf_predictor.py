"""
Real Predictor for Unknown Test Data: MyBlueBerryNights Hackamecum Competition
Check with Test Set provided with Known (Labelled) Date

You can run this example as follows:
    $ python <filename>.py
"""

import sys
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

def main():
    args = sys.argv[1:]

    if len(args)<1:
        print("MyBlueberry predictor needs some arguments")
        print("--simulate: if want to simulate behaviour with train set")
        print("--filename filename: if predictions for real test set are needed")
    else:
        simulate = False
        train_data = pd.read_excel("./datamecum_data/entrenamiento.xlsx")
        
        if args[0]=="--simulate":
            print("### Simulating Behaviour With All Training Data")
            print("### Training Data Has Been Loaded")            
            simulate = True
        
        elif args[0]=="--filename":                        
            test_data = pd.read_excel(str(args[1]))
            print(r"### Real Test Data Has Been Loaded ###")
                
        # Data Preparation
        data_dict = {}
        data_dict["data"] = train_data.iloc[:,:-1]
        data_dict["target"] = train_data.iloc[:,-1] # Train Data is Needed For scale_transform() + clustering
        if not simulate:
            data_dict["test"] = test_data
        
        test_x = bb_dataset(data_dict, simulate).scale_transform(ct,features_reduced_set).clustering().X_test
        if simulate:
            test_y = bb_dataset(data_dict, simulate).scale_transform(ct,features_reduced_set).clustering().y_test
        
        print(test_x.shape, test_y.shape)        
    return

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

    main()
    # Substitute with Real Test Competition Data
    # data = pd.read_excel("./datamecum_data/entrenamiento.xlsx")
    """ data = pd.DataFrame()

    data_dict["data"] = data """

    # Unknown a priori on Test Competition Set
    # data_dict["target"] = data.iloc[:,-1].map({"A": 0, "B": 1})

    