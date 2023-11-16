import numpy as np
import matplotlib.pyplot as plt
'\nAuthor:\n\tJack Cui\nBlog:\n    http://blog.csdn.net/c406495762\nZhihu:\n    https://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-10-10\n'

def loadDataSet(fileName):
    if False:
        print('Hello World!')
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return (dataMat, labelMat)

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    if False:
        i = 10
        return i + 15
    '\n\t单层决策树分类函数\n\tParameters:\n\t\tdataMatrix - 数据矩阵\n\t\tdimen - 第dimen列，也就是第几个特征\n\t\tthreshVal - 阈值\n\t\tthreshIneq - 标志\n\tReturns:\n\t\tretArray - 分类结果\n\t'
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    if False:
        print('Hello World!')
    '\n\t找到数据集上最佳的单层决策树\n\tParameters:\n\t\tdataArr - 数据矩阵\n\t\tclassLabels - 数据标签\n\t\tD - 样本权重\n\tReturns:\n\t\tbestStump - 最佳单层决策树信息\n\t\tminError - 最小误差\n\t\tbestClasEst - 最佳的分类结果\n\t'
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    (m, n) = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = float('inf')
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + float(j) * stepSize
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return (bestStump, minError, bestClasEst)

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    if False:
        for i in range(10):
            print('nop')
    '\n\t使用AdaBoost算法提升弱分类器性能\n\tParameters:\n\t\tdataArr - 数据矩阵\n\t\tclassLabels - 数据标签\n\t\tnumIt - 最大迭代次数\n\tReturns:\n\t\tweakClassArr - 训练好的分类器\n\t\taggClassEst - 类别估计累计值\n\t'
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        (bestStump, error, classEst) = buildStump(dataArr, classLabels, D)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        if errorRate == 0.0:
            break
    return (weakClassArr, aggClassEst)

def adaClassify(datToClass, classifierArr):
    if False:
        for i in range(10):
            print('nop')
    '\n\tAdaBoost分类函数\n\tParameters:\n\t\tdatToClass - 待分类样例\n\t\tclassifierArr - 训练好的分类器\n\tReturns:\n\t\t分类结果\n\t'
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return np.sign(aggClassEst)
if __name__ == '__main__':
    (dataArr, LabelArr) = loadDataSet('horseColicTraining2.txt')
    (weakClassArr, aggClassEst) = adaBoostTrainDS(dataArr, LabelArr)
    (testArr, testLabelArr) = loadDataSet('horseColicTest2.txt')
    print(weakClassArr)
    predictions = adaClassify(dataArr, weakClassArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != np.mat(LabelArr).T].sum() / len(dataArr) * 100))
    predictions = adaClassify(testArr, weakClassArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))