from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
NUMERICAL_COLS = ()
CATEGORICAL_COLS = np.arange(22) + 1

class DataTransformer:

    def fit(self, df):
        if False:
            while True:
                i = 10
        self.labelEncoders = {}
        self.scalers = {}
        for col in NUMERICAL_COLS:
            scaler = StandardScaler()
            scaler.fit(df[col].reshape(-1, 1))
            self.scalers[col] = scaler
        for col in CATEGORICAL_COLS:
            encoder = LabelEncoder()
            values = df[col].tolist()
            values.append('missing')
            encoder.fit(values)
            self.labelEncoders[col] = encoder
        self.D = len(NUMERICAL_COLS)
        for (col, encoder) in iteritems(self.labelEncoders):
            self.D += len(encoder.classes_)
        print('dimensionality:', self.D)

    def transform(self, df):
        if False:
            return 10
        (N, _) = df.shape
        X = np.zeros((N, self.D))
        i = 0
        for (col, scaler) in iteritems(self.scalers):
            X[:, i] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
            i += 1
        for (col, encoder) in iteritems(self.labelEncoders):
            K = len(encoder.classes_)
            X[np.arange(N), encoder.transform(df[col]) + i] = 1
            i += K
        return X

    def fit_transform(self, df):
        if False:
            for i in range(10):
                print('nop')
        self.fit(df)
        return self.transform(df)

def replace_missing(df):
    if False:
        while True:
            i = 10
    for col in NUMERICAL_COLS:
        if np.any(df[col].isnull()):
            med = np.median(df[col][df[col].notnull()])
            df.loc[df[col].isnull(), col] = med
    for col in CATEGORICAL_COLS:
        if np.any(df[col].isnull()):
            print(col)
            df.loc[df[col].isnull(), col] = 'missing'

def get_data():
    if False:
        for i in range(10):
            print('nop')
    df = pd.read_csv('../large_files/mushroom.data', header=None)
    df[0] = df.apply(lambda row: 0 if row[0] == 'e' else 1, axis=1)
    replace_missing(df)
    transformer = DataTransformer()
    X = transformer.fit_transform(df)
    Y = df[0].values
    return (X, Y)
if __name__ == '__main__':
    (X, Y) = get_data()
    baseline = LogisticRegression()
    print('CV baseline:', cross_val_score(baseline, X, Y, cv=8).mean())
    tree = DecisionTreeClassifier()
    print('CV one tree:', cross_val_score(tree, X, Y, cv=8).mean())
    model = RandomForestClassifier(n_estimators=20)
    print('CV forest:', cross_val_score(model, X, Y, cv=8).mean())