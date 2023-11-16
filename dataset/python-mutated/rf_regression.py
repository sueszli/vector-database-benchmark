from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
NUMERICAL_COLS = ['crim', 'zn', 'nonretail', 'nox', 'rooms', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
NO_TRANSFORM = ['river']

class DataTransformer:

    def fit(self, df):
        if False:
            print('Hello World!')
        self.scalers = {}
        for col in NUMERICAL_COLS:
            scaler = StandardScaler()
            scaler.fit(df[col].values.reshape(-1, 1))
            self.scalers[col] = scaler

    def transform(self, df):
        if False:
            print('Hello World!')
        (N, _) = df.shape
        D = len(NUMERICAL_COLS) + len(NO_TRANSFORM)
        X = np.zeros((N, D))
        i = 0
        for (col, scaler) in iteritems(self.scalers):
            X[:, i] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
            i += 1
        for col in NO_TRANSFORM:
            X[:, i] = df[col]
            i += 1
        return X

    def fit_transform(self, df):
        if False:
            for i in range(10):
                print('nop')
        self.fit(df)
        return self.transform(df)

def get_data():
    if False:
        while True:
            i = 10
    df = pd.read_csv('housing.data', header=None, delim_whitespace=True)
    df.columns = ['crim', 'zn', 'nonretail', 'river', 'nox', 'rooms', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']
    transformer = DataTransformer()
    N = len(df)
    train_idx = np.random.choice(N, size=int(0.7 * N), replace=False)
    test_idx = [i for i in range(N) if i not in train_idx]
    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]
    Xtrain = transformer.fit_transform(df_train)
    Ytrain = np.log(df_train['medv'].values)
    Xtest = transformer.transform(df_test)
    Ytest = np.log(df_test['medv'].values)
    return (Xtrain, Ytrain, Xtest, Ytest)
if __name__ == '__main__':
    (Xtrain, Ytrain, Xtest, Ytest) = get_data()
    model = RandomForestRegressor(n_estimators=100)
    model.fit(Xtrain, Ytrain)
    predictions = model.predict(Xtest)
    plt.scatter(Ytest, predictions)
    plt.xlabel('target')
    plt.ylabel('prediction')
    ymin = np.round(min(min(Ytest), min(predictions)))
    ymax = np.ceil(max(max(Ytest), max(predictions)))
    print('ymin:', ymin, 'ymax:', ymax)
    r = range(int(ymin), int(ymax) + 1)
    plt.plot(r, r)
    plt.show()
    plt.plot(Ytest, label='targets')
    plt.plot(predictions, label='predictions')
    plt.legend()
    plt.show()
    baseline = LinearRegression()
    single_tree = DecisionTreeRegressor()
    print('CV single tree:', cross_val_score(single_tree, Xtrain, Ytrain, cv=5).mean())
    print('CV baseline:', cross_val_score(baseline, Xtrain, Ytrain, cv=5).mean())
    print('CV forest:', cross_val_score(model, Xtrain, Ytrain, cv=5).mean())
    single_tree.fit(Xtrain, Ytrain)
    baseline.fit(Xtrain, Ytrain)
    print('test score single tree:', single_tree.score(Xtest, Ytest))
    print('test score baseline:', baseline.score(Xtest, Ytest))
    print('test score forest:', model.score(Xtest, Ytest))