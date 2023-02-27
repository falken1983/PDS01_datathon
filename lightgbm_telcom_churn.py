"""
Optuna template that optimizes a classifier configuration for telcom churn using LightGBM.

In this example, we optimize the validation accuracy of telcom churn using LightGBM.
We optimize both the choice of booster model and their hyperparameters.

"""
import os

import pandas as pd
import numpy as np
import optuna

import lightgbm as lgb
#import sklearn.datasets
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics

# optuna logging
from datetime import datetime
import joblib


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
THISFILE_DIRNAME = os.path.dirname(os.path.realpath(__file__))
#CLEAN_DATA_REALPATH = THISFILE_DIRNAME + "/" + "kaggle_data/cleaned_train.csv"
CLEAN_DATA_REALPATH = THISFILE_DIRNAME + "/" + "datamecum_data/cleaned_telcom_churn_construction.csv"
OPTUNA_RESULTS_REALPATH = THISFILE_DIRNAME + "/" + "optuna_runs"

def load_cleaned_telcom_churn(path=CLEAN_DATA_REALPATH):
    data_raw = pd.read_csv(path)    
    X = data_raw.iloc[:,:-1]
    y = data_raw[data_raw.columns.tolist()[-1]]
    return X,y

def scaler(data, num_todrop=["network_age","spend_mo12"], num_topass=["complaint_calls"]):    
    numerical_features = data.select_dtypes("number").columns.tolist()
    cat_features = data.select_dtypes("object").columns.tolist()
    
    excluded = list(set(num_todrop).union(set(num_topass)))
    num_totransform = [f for f in numerical_features if f not in excluded]
 
    ct = ColumnTransformer(
        [
            ("id_drop", "drop", cat_features[0]),
            ("num_drop", "drop", num_todrop),
            ("num_transform", PowerTransformer(), num_totransform),
            ("num_pass", "passthrough", num_topass),
            ("cat_drop", "drop", cat_features[1:])
        ]
    )

    return ct



def objective(trial):
    # Data, Target Definition 
    # data, target = load_cleaned_telcom_churn()    
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25, random_state=42)

    scl = scaler(train_x)
    train_x = scl.fit_transform(train_x)
    valid_x = scl.transform(valid_x)
    dtrain = lgb.Dataset(train_x, label=train_y)

    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        #"device_type": trial.suggest_categorical("device_type", ['gpu']),
        #"n_estimators": trial.suggest_int("n_estimators", 10,100, step=10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "max_bin": trial.suggest_int("max_bin", 200, 300),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-7, 10, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-7, 10, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.05),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.05),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    }

    """ 
                 "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),

       """

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    data, target = load_cleaned_telcom_churn()
    print(f"{CLEAN_DATA_REALPATH} succesfully loaded and parsed to Pandas objects\n")

    study.optimize(objective, n_trials=500)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # logging
    the_time_is_now = datetime.now().strftime("%Y%m%d_%H%M")
    filename_dump = f"{OPTUNA_RESULTS_REALPATH}" + "/" + f"{the_time_is_now}_lgb_optuna_study_batch.pkl" 
    joblib.dump(study, filename=filename_dump)