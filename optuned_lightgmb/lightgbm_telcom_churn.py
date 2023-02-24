"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM.

In this example, we optimize the validation accuracy of cancer detection using LightGBM.
We optimize both the choice of booster model and their hyperparameters.

"""
import pandas as pd
import numpy as np
import optuna

import lightgbm as lgb
#import sklearn.datasets
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics

# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).

def load_cleaned_telcom_churn(path="./cleaned_telcom_churn_construction.csv"):
    data_raw = pd.read_csv(path)
    print(f"{path} succesfully loaded\n")
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
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    data, target = load_cleaned_telcom_churn("../kaggle_data/cleaned_train.csv")

    study.optimize(objective, n_trials=500)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
