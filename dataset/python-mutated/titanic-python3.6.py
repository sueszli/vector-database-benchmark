"""
Created on 2019-08-14
Update  on 2019-08-31
Author: 片刻
Github: https://github.com/apachecn/Interview
"""
import re
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier

def opencsv():
    if False:
        i = 10
        return i + 15
    root_path = '/opt/data/kaggle/getting-started/titanic'
    tr_data = pd.read_csv('%s/%s' % (root_path, 'input/train.csv'), header=0)
    te_data = pd.read_csv('%s/%s' % (root_path, 'input/test.csv'), header=0)
    do_DataPreprocessing(tr_data)
    do_DataPreprocessing(te_data)
    pids = te_data['PassengerId'].tolist()
    tr_data.drop(['PassengerId'], axis=1, inplace=True)
    te_data.drop(['PassengerId'], axis=1, inplace=True)
    train_data = tr_data.values[:, 1:]
    train_label = tr_data.values[:, 0]
    test_data = te_data.values[:, :]
    return (train_data, train_label, test_data, pids)

def do_DataPreprocessing(titanic):
    if False:
        for i in range(10):
            print('nop')
    '\n    | Survival    | 生存                | 0 = No, 1 = Yes |\n    | Pclass      | 票类别-社会地位       | 1 = 1st, 2 = 2nd, 3 = 3rd |  \n    | Name        | 姓名                | |\n    | Sex         | 性别                | |\n    | Age         | 年龄                | |    \n    | SibSp       | 船上的兄弟姐妹/配偶   | | \n    | Parch       | 船上的父母/孩子的数量 | |\n    | Ticket      | 票号                | |   \n    | Fare        | 乘客票价            | |  \n    | Cabin       | 客舱号码            | |    \n    | Embarked    | 登船港口            | C=Cherbourg, Q=Queenstown, S=Southampton |  \n\n    >>> print(titanic.describe())\n           PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare\n    count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000\n    mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208\n    std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429\n    min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000\n    25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400\n    50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200\n    75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000\n    max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200\n\n    Pclass  Name                          Sex     Age      SibSp   Parch   Ticket      Fare        Cabin   Embarked\n    3       Braund, Mr. Owen Harris       male    22       1       0       A/5 21171   7.25                S\n    1       Cumings, Mrs. John Bradley    female  38       1       0       PC 17599    71.2833     C85     C\n    '
    titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
    titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].median())
    titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
    titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
    '\n    titanic[["Embarked"]].groupby("Embarked").agg({"Embarked": "count"})\n              Embarked\n    Embarked          \n    C              168\n    Q               77\n    S              644\n    '
    titanic['Embarked'] = titanic['Embarked'].fillna('S')
    titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
    titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
    titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2

    def get_title(name):
        if False:
            return 10
        title_search = re.search(' ([A-Za-z]+)\\.', name)
        if title_search:
            return title_search.group(1)
        return ''
    titles = titanic['Name'].apply(get_title)
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Major': 7, 'Col': 7, 'Mlle': 8, 'Mme': 8, 'Don': 9, 'Dona': 9, 'Lady': 10, 'Countess': 10, 'Jonkheer': 10, 'Sir': 9, 'Capt': 7, 'Ms': 2}
    for (k, v) in title_mapping.items():
        titles[titles == k] = v
    titanic['Title'] = [int(i) for i in titles.values.tolist()]
    titanic['NameLength'] = titanic['Name'].apply(lambda x: len(x))
    titanic.drop(['Cabin'], axis=1, inplace=True)
    titanic.drop(['SibSp'], axis=1, inplace=True)
    titanic.drop(['Ticket'], axis=1, inplace=True)
    titanic.drop(['Name'], axis=1, inplace=True)

def do_FeatureEngineering(data, COMPONENT_NUM=0.9):
    if False:
        i = 10
        return i + 15
    scaler = preprocessing.StandardScaler()
    s_data = scaler.fit_transform(data)
    return s_data

def trainModel(trainData, trainLabel):
    if False:
        i = 10
        return i + 15
    print('模型融合')
    '\n    Bagging:   同一模型的投票选举\n    Boosting:  同一模型的再学习\n    Voting:    不同模型的投票选举\n    Stacking:  分层预测 – K-1份数据预测1份模型拼接，得到 预测结果*算法数（作为特征） => 从而预测最终结果\n    Blending:  分层预测 – 将数据分成2部分（A部分训练B部分得到预测结果），得到 预测结果*算法数（作为特征） => 从而预测最终结果\n    '
    clfs = [AdaBoostClassifier(), SVC(probability=True), AdaBoostClassifier(), LogisticRegression(C=0.1, max_iter=100), XGBClassifier(max_depth=6, n_estimators=100, num_round=5), RandomForestClassifier(n_estimators=100, max_depth=6, oob_score=True), GradientBoostingClassifier(learning_rate=0.3, max_depth=6, n_estimators=100)]
    (X_d1, X_d2, y_d1, y_d2) = train_test_split(trainData, trainLabel, test_size=0.5, random_state=2017)
    dataset_d1 = np.zeros((X_d2.shape[0], len(clfs)))
    dataset_d2 = np.zeros((trainLabel.shape[0], len(clfs)))
    for (j, clf) in enumerate(clfs):
        clf.fit(X_d1, y_d1)
        dataset_d1[:, j] = clf.predict_proba(X_d2)[:, 1]
    model = LogisticRegression(C=0.1, max_iter=100)
    model.fit(dataset_d1, y_d2)
    scores = cross_val_score(model, dataset_d1, y_d2, cv=5, scoring='roc_auc')
    print(scores.mean(), '\n', scores)
    return model

def main():
    if False:
        print('Hello World!')
    sta_time = datetime.datetime.now()
    (train_data, train_label, test_data, pids) = opencsv()
    pca_tr_data = do_FeatureEngineering(train_data)
    pca_te_data = do_FeatureEngineering(test_data)
    model = trainModel(pca_tr_data, train_label)
    model.fit(pca_tr_data, train_label)
    labels = model.predict(pca_te_data)
    print(type(pids), type(labels.tolist()))
    result = pd.DataFrame({'PassengerId': pids, 'Survived': [int(i) for i in labels.tolist()]})
    result.to_csv('Result_titanic.csv', index=False)
    end_time = datetime.datetime.now()
    times = (end_time - sta_time).seconds
    print('\n运行时间: %ss == %sm == %sh\n\n' % (times, times / 60, times / 60 / 60))
if __name__ == '__main__':
    main()