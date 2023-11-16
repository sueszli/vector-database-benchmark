"""
Created on 2017-12-4 15:45:02
@Team: 瑶瑶亲卫队
"""
import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
train = pd.read_csv('D:/titanic/titanic_dataset/train.csv')
test = pd.read_csv('D:/titanic/titanic_dataset/test.csv')
PassengerId = test['PassengerId']
full_data = [train, test]
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)

def get_title(name):
    if False:
        return 10
    title_search = re.search(' ([A-Za-z]+)\\.', name)
    if title_search:
        return title_search.group(1)
    return ''
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

def saveTmpTrainFile(tmpFile, csvName):
    if False:
        while True:
            i = 10
    with open(csvName, 'w', newline='') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(['Survived', 'Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked', 'Name_length', 'Has_cabin', 'FamilySize', 'IsAlone', 'Title'])
        for lines in tmpFile.index:
            tmp = []
            tmp.append(tmpFile.loc[lines].values[1])
            tmp.append(tmpFile.loc[lines].values[2])
            tmp.append(tmpFile.loc[lines].values[4])
            tmp.append(tmpFile.loc[lines].values[5])
            tmp.append(tmpFile.loc[lines].values[7])
            tmp.append(tmpFile.loc[lines].values[9])
            tmp.append(tmpFile.loc[lines].values[11])
            tmp.append(tmpFile.loc[lines].values[12])
            tmp.append(tmpFile.loc[lines].values[13])
            tmp.append(tmpFile.loc[lines].values[14])
            tmp.append(tmpFile.loc[lines].values[15])
            tmp.append(tmpFile.loc[lines].values[-1])
            myWriter.writerow(tmp)
saveTmpTrainFile(train, 'D:/titanic/titanic_dataset/train_later.csv')

def saveTmpTestFile(tmpFile, csvName):
    if False:
        return 10
    with open(csvName, 'w', newline='') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked', 'Name_length', 'Has_cabin', 'FamilySize', 'IsAlone', 'Title'])
        for lines in tmpFile.index:
            tmp = []
            tmp.append(tmpFile.loc[lines].values[1])
            tmp.append(tmpFile.loc[lines].values[3])
            tmp.append(tmpFile.loc[lines].values[4])
            tmp.append(tmpFile.loc[lines].values[6])
            tmp.append(tmpFile.loc[lines].values[8])
            tmp.append(tmpFile.loc[lines].values[10])
            tmp.append(tmpFile.loc[lines].values[11])
            tmp.append(tmpFile.loc[lines].values[12])
            tmp.append(tmpFile.loc[lines].values[13])
            tmp.append(tmpFile.loc[lines].values[14])
            tmp.append(tmpFile.loc[lines].values[15])
            myWriter.writerow(tmp)
saveTmpFile(test, 'D:/titanic/titanic_dataset/test_later.csv')
train_later = pd.read_csv('D:/titanic/titanic_dataset/train_later.csv')
test_later = pd.read_csv('D:/titanic/titanic_dataset/test_later.csv')
ntrain = train_later.shape[0]
ntest = test_later.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):
        if False:
            i = 10
            return i + 15
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        if False:
            while True:
                i = 10
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        if False:
            print('Hello World!')
        return self.clf.predict(x)

    def fit(self, x, y):
        if False:
            while True:
                i = 10
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        if False:
            while True:
                i = 10
        print(self.clf.fit(x, y).feature_importances_)

def get_oof(clf, x_train, y_train, x_test):
    if False:
        return 10
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    for (i, (train_index, test_index)) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        clf.train(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return (oof_train.reshape(-1, 1), oof_test.reshape(-1, 1))
rf_params = {'n_jobs': -1, 'n_estimators': 500, 'warm_start': True, 'max_depth': 6, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'verbose': 0}
et_params = {'n_jobs': -1, 'n_estimators': 500, 'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0}
ada_params = {'n_estimators': 500, 'learning_rate': 0.75}
gb_params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_leaf': 2, 'verbose': 0}
svc_params = {'kernel': 'linear', 'C': 0.025}
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values
x_test = test.values
(et_oof_train, et_oof_test) = get_oof(et, x_train, y_train, x_test)
(rf_oof_train, rf_oof_test) = get_oof(rf, x_train, y_train, x_test)
(ada_oof_train, ada_oof_test) = get_oof(ada, x_train, y_train, x_test)
(gb_oof_train, gb_oof_test) = get_oof(gb, x_train, y_train, x_test)
(svc_oof_train, svc_oof_test) = get_oof(svc, x_train, y_train, x_test)
print('Training is complete')

def saveResult(result, csvName):
    if False:
        print('Hello World!')
    with open(csvName, 'w', newline='') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(['PassengerId', 'Survived'])
        index = 891
        for i in result:
            tmp = []
            index = index + 1
            tmp.append(index)
            tmp.append(int(i))
            myWriter.writerow(tmp)
saveResult(et_oof_test, 'D:/titanic/titanic_dataset/result/et.csv')
saveResult(rf_oof_test, 'D:/titanic/titanic_dataset/result/rf.csv')
saveResult(ada_oof_test, 'D:/titanic/titanic_dataset/result/ada.csv')
saveResult(gb_oof_test, 'D:/titanic/titanic_dataset/result/gb.csv')
saveResult(svc_oof_test, 'D:/titanic/titanic_dataset/result/svc.csv')