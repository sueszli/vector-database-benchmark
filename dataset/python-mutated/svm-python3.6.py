"""
Created on 2017-10-26
Update  on 2017-10-26
Author: 片刻
Github: https://github.com/apachecn/kaggle
"""
import os
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
data_dir = '/opt/data/kaggle/getting-started/digit-recognizer/'

def opencsv():
    if False:
        print('Hello World!')
    print('Load Data...')
    dataTrain = pd.read_csv(os.path.join(data_dir, 'input/train.csv'))
    dataPre = pd.read_csv(os.path.join(data_dir, 'input/test.csv'))
    trainData = dataTrain.values[:, 1:]
    trainLabel = dataTrain.values[:, 0]
    preData = dataPre.values[:, :]
    return (trainData, trainLabel, preData)

def dRCsv(x_train, x_test, preData, COMPONENT_NUM):
    if False:
        for i in range(10):
            print('nop')
    print('dimensionality reduction...')
    trainData = np.array(x_train)
    testData = np.array(x_test)
    preData = np.array(preData)
    '\n    使用说明：https://www.cnblogs.com/pinard/p/6243025.html\n    n_components>=1\n      n_components=NUM  \x08 设置\x08占特征数量比\n    0 < n_components < 1\n      n_components=0.99  \x08设置阈值总方差占比\n    '
    pca = PCA(n_components=COMPONENT_NUM, whiten=True)
    pca.fit(trainData)
    pcaTrainData = pca.transform(trainData)
    pcaTestData = pca.transform(testData)
    pcaPreData = pca.transform(preData)
    print(pca.explained_variance_, '\n', pca.explained_variance_ratio_, '\n', pca.n_components_)
    print(sum(pca.explained_variance_ratio_))
    return (pcaTrainData, pcaTestData, pcaPreData)

def trainModel(trainData, trainLabel):
    if False:
        print('Hello World!')
    print('Train SVM...')
    clf = SVC(C=4, kernel='rbf')
    clf.fit(trainData, trainLabel)
    return clf

def saveResult(result, csvName):
    if False:
        print('Hello World!')
    with open(csvName, 'w') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(['ImageId', 'Label'])
        index = 0
        for r in result:
            index += 1
            myWriter.writerow([index, int(r)])
    print('Saved successfully...')

def analyse_data(dataMat):
    if False:
        for i in range(10):
            print('nop')
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    (eigvals, eigVects) = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigvals)
    topNfeat = 100
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    cov_all_score = float(sum(eigvals))
    sum_cov_score = 0
    for i in range(0, len(eigValInd)):
        line_cov_score = float(eigvals[eigValInd[i]])
        sum_cov_score += line_cov_score
        '\n        我们发现其中有超过20%的特征值都是0。\n        这就意味着这些特征都是其他特征的副本，也就是说，它们可以通过其他特征来表示，而本身并没有提供额外的信息。\n\n        最前面15个值的数量级大于10^5，实际上那以后的值都变得非常小。\n        这就相当于告诉我们只有部分重要特征，重要特征的数目也很快就会下降。\n\n        最后，我们可能会注意到有一些小的负值，他们主要源自数值误差应该四舍五入成0.\n        '
        print('主成分：%s, 方差占比：%s%%, 累积方差占比：%s%%' % (format(i + 1, '2.0f'), format(line_cov_score / cov_all_score * 100, '4.2f'), format(sum_cov_score / cov_all_score * 100, '4.1f')))

def getOptimalAccuracy(trainData, trainLabel, preData):
    if False:
        for i in range(10):
            print('nop')
    (x_train, x_test, y_train, y_test) = train_test_split(trainData, trainLabel, test_size=0.1)
    (lineLen, featureLen) = np.shape(x_test)
    minErr = 1
    minSumErr = 0
    optimalNum = 1
    optimalLabel = []
    optimalSVMClf = None
    pcaPreDataResult = None
    for i in range(30, 45, 1):
        (pcaTrainData, pcaTestData, pcaPreData) = dRCsv(x_train, x_test, preData, i)
        clf = trainModel(pcaTrainData, y_train)
        testLabel = clf.predict(pcaTestData)
        errArr = np.mat(np.ones((lineLen, 1)))
        sumErrArr = errArr[testLabel != y_test].sum()
        sumErr = sumErrArr / lineLen
        print('i=%s' % i, lineLen, sumErrArr, sumErr)
        if sumErr <= minErr:
            minErr = sumErr
            minSumErr = sumErrArr
            optimalNum = i
            optimalSVMClf = clf
            optimalLabel = testLabel
            pcaPreDataResult = pcaPreData
            print('i=%s >>>>> \t' % i, lineLen, int(minSumErr), 1 - minErr)
    '\n    展现 准确率与召回率\n        precision 准确率\n        recall 召回率\n        f1-score  准确率和召回率的一个综合得分\n        support 参与比较的数量\n    参考链接：http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report\n    '
    target_names = [str(i) for i in list(set(y_test))]
    print(target_names)
    print(classification_report(y_test, optimalLabel, target_names=target_names))
    print('特征数量= %s, 存在最优解：>>> \t' % optimalNum, lineLen, int(minSumErr), 1 - minErr)
    return (optimalSVMClf, pcaPreDataResult)

def storeModel(model, filename):
    if False:
        for i in range(10):
            print('nop')
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(model, fw)

def getModel(filename):
    if False:
        while True:
            i = 10
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

def trainDRSVM():
    if False:
        i = 10
        return i + 15
    startTime = time.time()
    (trainData, trainLabel, preData) = opencsv()
    (optimalSVMClf, pcaPreData) = getOptimalAccuracy(trainData, trainLabel, preData)
    storeModel(optimalSVMClf, os.path.join(data_dir, 'output/Result_sklearn_SVM.model'))
    storeModel(pcaPreData, os.path.join(data_dir, 'output/Result_sklearn_SVM.pcaPreData'))
    print('finish!')
    stopTime = time.time()
    print('TrainModel store time used:%f s' % (stopTime - startTime))

def preDRSVM():
    if False:
        print('Hello World!')
    startTime = time.time()
    optimalSVMClf = getModel(os.path.join(data_dir, 'output/Result_sklearn_SVM.model'))
    pcaPreData = getModel(os.path.join(data_dir, 'output/Result_sklearn_SVM.pcaPreData'))
    testLabel = optimalSVMClf.predict(pcaPreData)
    saveResult(testLabel, os.path.join(data_dir, 'output/Result_sklearn_SVM.csv'))
    print('finish!')
    stopTime = time.time()
    print('PreModel load time used:%f s' % (stopTime - startTime))

def dataVisulization(data, labels):
    if False:
        return 10
    pca = PCA(n_components=2, whiten=True)
    pca.fit(data)
    pcaData = pca.transform(data)
    uniqueClasses = set(labels)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for cClass in uniqueClasses:
        plt.scatter(pcaData[labels == cClass, 0], pcaData[labels == cClass, 1])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('MNIST visualization')
    plt.show()
if __name__ == '__main__':
    (trainData, trainLabel, preData) = opencsv()
    dataVisulization(trainData, trainLabel)