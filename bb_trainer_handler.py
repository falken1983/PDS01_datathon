import numpy as np
import pandas as pd
#from sklearnsets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score

class bb_dataset():
    def _load_data(self, sklearn_load_ds, simulated=True):
        data = sklearn_load_ds
        X = data["data"]
        if simulated:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, data["target"], test_size=0.3, random_state=42)       
        else:            
            self.X_train, self.y_train = data["data"], data["target"] # <- Train Data
            self.X_test = data["test"]                 # <- Real Test Data

        
    def __init__(self, sklearn_load_ds, sim=True):
        self._load_data(sklearn_load_ds, sim)
    
    def scale_transform(self, transformer, f_names):
        feature_names = f_names
        self.X_train = transformer.fit_transform(self.X_train)
        self.X_test = transformer.transform(self.X_test)
        self.X_train = pd.DataFrame(self.X_train, columns=feature_names)
        self.X_test = pd.DataFrame(self.X_test, columns=feature_names)
        return self
    
    def classify(self, model=LogisticRegression(random_state=42)):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print('Accuracy: {}'.format(accuracy_score(self.y_test, y_pred)))
        return model

    def gs_CV(self, search_CV):
        search_CV.fit(self.X_train, self.y_train)
        return search_CV

    def clustering(self, output='add'):
        n_clusters = len(np.unique(self.y_train))
        clf = KMeans(n_clusters = n_clusters, random_state=42)
        # clf = AgglomerativeClustering(n_clusters = n_clusters)
        clf.fit(self.X_train)
        y_labels_train = clf.labels_
        # y_labels_test = clf.fit_predict(self.X_test)
        y_labels_test = clf.predict(self.X_test)
        if output == 'add':
            self.X_train['km_clust'] = y_labels_train
            self.X_test['km_clust'] = y_labels_test
        elif output == 'replace':
            self.X_train = y_labels_train[:, np.newaxis]
            self.X_test = y_labels_test[:, np.newaxis]
        else:
            raise ValueError('output should be either add or replace')
        return self
