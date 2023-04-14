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

from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

def best_estimator_loader():
    with open("./best_model.pkl","rb") as fmodel:
        return pickle.load(fmodel)

def best_estimator_predictions(train_x, train_y, test_x):
    # Load Best Estimator
    best_estimator = best_estimator_loader()
    
    # Refitting (with all Labelled Data)
    best_estimator.fit(train_x, train_y)
    
    # Predicting Labels (with unknown or simulated test data) and improving acc/AUC with trained treshold.
    with open("./best_model_features.yaml","r") as f:
        specs = yaml.safe_load(f)
    th_proba = specs["ROC"]
    label_scores = best_estimator.predict_proba(test_x)[:,1]
    return label_scores, np.where(label_scores>th_proba,1,0)

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
            print("### Real Test Data Has Been Loaded ###")
            print(f"### Number Of Samples: {test_data.shape[0]}")
            features = test_data.iloc[:,:-1].columns.tolist()
            print(f"### Number Of Features: {len(features)}")
            print(f"### Features: {features}")
        # Data Preparation
        data_dict = {}
        data_dict["data"] = train_data.iloc[:,:-1]
        data_dict["target"] = train_data.iloc[:,-1] # Train Data is Needed For scale_transform() + clustering
        if not simulate:
            data_dict["test"] = test_data
        
        # (train_x, train_y) for refitting purposes
        train_x = bb_dataset(data_dict, simulate).scale_transform(ct,features_reduced_set).clustering().X_train
        train_y = bb_dataset(data_dict, simulate).scale_transform(ct,features_reduced_set).clustering().y_train
        
        # Real world data tranches for evaluating purposes
        test_x = bb_dataset(data_dict, simulate).scale_transform(ct,features_reduced_set).clustering().X_test
        
        if simulate:
            test_y = bb_dataset(data_dict, simulate).scale_transform(ct,features_reduced_set).clustering().y_test
        
        # Predictions
        scores, _ = best_estimator_predictions(train_x, train_y, test_x)        

        if simulate:
            df_predict = pd.DataFrame(data={
                                    "observed": test_y.map({"A": 0, "B": 1}),
                                    "proba_class_1": scores
                                    }
            )
            print(classification_report(df_predict["observed"], df_predict["predicted"]))
        else:
            df_predict = pd.DataFrame(data={"proba_class_1": scores})            
        
        test_x.to_csv("./features_transformed.csv", index=False)
        test_x.to_excel("./features_transformed.xlsx", index=False)

        df_predict.to_csv("./predictions_test.csv", index=False)        
        df_predict.to_excel("./predictions_test.xlsx", index=False)
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

    