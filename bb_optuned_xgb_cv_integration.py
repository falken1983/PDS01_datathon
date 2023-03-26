"""
Optuna example that demonstrates a pruner for XGBoost.cv.

In this example, we optimize the validation auc of cancer detection using XGBoost.
We optimize both the choice of booster model and their hyperparameters. Throughout
training of models, a pruner observes intermediate results and stop unpromising trials.

You can run this example as follows:
    $ python xgboost_cv_integration.py

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
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import StandardScaler, RobustScaler


import xgboost as xgb


def objective(trial):
    #train_x, train_y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x = bb_dataset(data_dict).scale_transform(ct, features_reduced_set).clustering().X_train
    train_y = bb_dataset(data_dict).scale_transform(ct, features_reduced_set).clustering().y_train

    dtrain = xgb.DMatrix(train_x, label=train_y)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")
    history = xgb.cv(param, dtrain, num_boost_round=100, callbacks=[pruning_callback])

    mean_auc = history["test-auc-mean"].values[-1]
    return mean_auc


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

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(pruner=pruner, direction="maximize")
    study.optimize(objective, n_trials=1000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
