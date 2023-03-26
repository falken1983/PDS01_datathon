"""
Optuna example that demonstrates a optimizer for RandomForest with vanilla cv.
Data: MyBlueBerryNights Hackamecum Competition

You can run this example as follows:
    $ python <filename>.py
"""
import optuna


""" For DataSet Preparation """
from bb_trainer_handler import bb_dataset
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # El filtrado de NAs hace la serie excesivamente  

import sklearn.ensemble
import sklearn.model_selection


def objective(trial):
    train_x = bb_dataset(data_dict).scale_transform(ct, features_reduced_set).clustering().X_train
    train_y = bb_dataset(data_dict).scale_transform(ct, features_reduced_set).clustering().y_train

    param = {
        "max_depth": int(trial.suggest_float("max_depth", 1, 32, log=True)),
        "n_estimators": trial.suggest_int("n_estimators", 2, 100),
        'min_samples_split': trial.suggest_int("min_samples_split", 3, 30),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 2, 10)
    }

    clf = sklearn.ensemble.RandomForestClassifier(**param)
    
    return sklearn.model_selection.cross_val_score(
        clf, train_x, train_y, n_jobs=-1, cv=5).mean()


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

    data = pd.read_excel("./datamecum_data/entrenamiento.xlsx")
    data_dict["data"] = data.iloc[:,:-1]
    data_dict["target"] = data.iloc[:,-1].map({"A": 0, "B": 1})

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1000)

    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))