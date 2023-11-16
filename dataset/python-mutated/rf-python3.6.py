"""
Created on 2018-05-14
Update  on 2018-05-19
Author: 平淡的天/wang-sw
Github: https://github.com/apachecn/kaggle
"""
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
data_dir = '/Users/wuyanxue/Documents/GitHub/datasets/getting-started/digit-recognizer/'

def opencsv():
    if False:
        while True:
            i = 10
    train_data = pd.read_csv(os.path.join(data_dir, 'input/train.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'input/test.csv'))
    data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
    data.drop(['label'], axis=1, inplace=True)
    label = train_data.label
    return (train_data, test_data, data, label)

def dRPCA(data, COMPONENT_NUM=100):
    if False:
        while True:
            i = 10
    print('dimensionality reduction...')
    data = np.array(data)
    '\n    使用说明：https://www.cnblogs.com/pinard/p/6243025.html\n    n_components>=1\n      n_components=NUM  \x08 设置\x08占特征数量\n    0 < n_components < 1\n      n_components=0.99  \x08设置阈值总方差占比\n    '
    pca = PCA(n_components=COMPONENT_NUM, random_state=34)
    data_pca = pca.fit_transform(data)
    print(pca.explained_variance_, '\n', pca.explained_variance_ratio_, '\n', pca.n_components_)
    print(sum(pca.explained_variance_ratio_))
    storeModel(data_pca, os.path.join(data_dir, 'output/Result_sklearn_rf.pcaData'))
    return data_pca

def trainModel(X_train, y_train):
    if False:
        while True:
            i = 10
    print('Train RF...')
    clf = RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=34)
    clf.fit(X_train, y_train)
    return clf

def printAccuracy(y_test, y_predict):
    if False:
        i = 10
        return i + 15
    zeroLable = y_test - y_predict
    rightCount = 0
    for i in range(len(zeroLable)):
        if list(zeroLable)[i] == 0:
            rightCount += 1
    print('the right rate is:', float(rightCount) / len(zeroLable))

def storeModel(model, filename):
    if False:
        while True:
            i = 10
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(model, fw)

def getModel(filename):
    if False:
        print('Hello World!')
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

def saveResult(result, csvName):
    if False:
        while True:
            i = 10
    i = 0
    n = len(result)
    print('the size of test set is {}'.format(n))
    with open(os.path.join(data_dir, 'output/Result_sklearn_RF.csv'), 'w') as fw:
        fw.write('{},{}\n'.format('ImageId', 'Label'))
        for i in range(1, n + 1):
            fw.write('{},{}\n'.format(i, result[i - 1]))
    print('Result saved successfully... and the path = {}'.format(csvName))

def trainRF():
    if False:
        print('Hello World!')
    start_time = time.time()
    (train_data, test_data, data, label) = opencsv()
    print('load data finish')
    stop_time_l = time.time()
    print('load data time used:%f s' % (stop_time_l - start_time))
    startTime = time.time()
    data_pca = dRPCA(data, 100)
    (X_train, X_test, y_train, y_test) = train_test_split(data_pca[0:len(train_data)], label, test_size=0.1, random_state=34)
    clf = trainModel(X_train, y_train)
    storeModel(data_pca[len(train_data):], os.path.join(data_dir, 'output/Result_sklearn_rf.pcaPreData'))
    storeModel(clf, os.path.join(data_dir, 'output/Result_sklearn_rf.model'))
    y_predict = clf.predict(X_test)
    printAccuracy(y_test, y_predict)
    print('finish!')
    stopTime = time.time()
    print('TrainModel store time used:%f s' % (stopTime - startTime))

def preRF():
    if False:
        for i in range(10):
            print('nop')
    startTime = time.time()
    clf = getModel(os.path.join(data_dir, 'output/Result_sklearn_rf.model'))
    pcaPreData = getModel(os.path.join(data_dir, 'output/Result_sklearn_rf.pcaPreData'))
    result = clf.predict(pcaPreData)
    saveResult(result, os.path.join(data_dir, 'output/Result_sklearn_rf.csv'))
    print('finish!')
    stopTime = time.time()
    print('PreModel load time used:%f s' % (stopTime - startTime))
if __name__ == '__main__':
    trainRF()
    preRF()