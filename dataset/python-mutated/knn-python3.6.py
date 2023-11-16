"""
Created on 2017-10-26
Update  on 2018-05-16
Author: 片刻/ccyf00
Github: https://github.com/apachecn/kaggle
"""
import os
import csv
import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
data_dir = '/opt/data/kaggle/getting-started/digit-recognizer/'

def opencsv():
    if False:
        i = 10
        return i + 15
    data = pd.read_csv(os.path.join(data_dir, 'input/train.csv'))
    data1 = pd.read_csv(os.path.join(data_dir, 'input/test.csv'))
    train_data = data.values[:, 1:]
    train_label = data.values[:, 0]
    test_data = data1.values[:, :]
    return (train_data, train_label, test_data)

def dRPCA(x_train, x_test, COMPONENT_NUM):
    if False:
        for i in range(10):
            print('nop')
    print('dimensionality reduction...')
    trainData = np.array(x_train)
    testData = np.array(x_test)
    '\n    使用说明：https://www.cnblogs.com/pinard/p/6243025.html\n    n_components>=1\n      n_components=NUM  \x08 设置\x08占特征数量比\n    0 < n_components < 1\n      n_components=0.99  \x08设置阈值总方差占比\n    '
    pca = PCA(n_components=COMPONENT_NUM, whiten=False)
    pca.fit(trainData)
    pcaTrainData = pca.transform(trainData)
    pcaTestData = pca.transform(testData)
    print('特征数量: %s' % pca.n_components_)
    print('总方差占比: %s' % sum(pca.explained_variance_ratio_))
    return (pcaTrainData, pcaTestData)

def trainModel(trainData, trainLabel):
    if False:
        i = 10
        return i + 15
    clf = KNeighborsClassifier()
    clf.fit(trainData, np.ravel(trainLabel))
    return clf

def saveResult(result, csvName):
    if False:
        i = 10
        return i + 15
    with open(csvName, 'w') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(['ImageId', 'Label'])
        index = 0
        for r in result:
            index += 1
            myWriter.writerow([index, int(r)])
    print('Saved successfully...')

def dRecognition_knn():
    if False:
        for i in range(10):
            print('nop')
    sta_time = datetime.datetime.now()
    (trainData, trainLabel, testData) = opencsv()
    print('load data finish')
    end_time_1 = datetime.datetime.now()
    print('load data time used: %s' % end_time_1)
    (trainDataPCA, testDataPCA) = dRPCA(trainData, testData, 0.8)
    clf = trainModel(trainDataPCA, trainLabel)
    testLabel = clf.predict(testDataPCA)
    saveResult(testLabel, os.path.join(data_dir, 'output/Result_knn.csv'))
    print('finish!')
    end_time = datetime.datetime.now()
    times = (end_time - sta_time).seconds
    print('\n运行时间: %ss == %sm == %sh\n\n' % (times, times / 60, times / 60 / 60))
if __name__ == '__main__':
    dRecognition_knn()