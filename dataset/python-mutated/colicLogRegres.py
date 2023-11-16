from sklearn.linear_model import LogisticRegression
import numpy as np
import random
'\n函数说明:sigmoid函数\n\nParameters:\n\tinX - 数据\nReturns:\n\tsigmoid函数\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-09-05\n'

def sigmoid(inX):
    if False:
        i = 10
        return i + 15
    return 1.0 / (1 + np.exp(-inX))
'\n函数说明:改进的随机梯度上升算法\n\nParameters:\n\tdataMatrix - 数据数组\n\tclassLabels - 数据标签\n\tnumIter - 迭代次数\nReturns:\n\tweights - 求得的回归系数数组(最优参数)\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-09-05\n'

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    if False:
        for i in range(10):
            print('nop')
    (m, n) = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del dataIndex[randIndex]
    return weights
'\n函数说明:梯度上升算法\n\nParameters:\n\tdataMatIn - 数据集\n\tclassLabels - 数据标签\nReturns:\n\tweights.getA() - 求得的权重数组(最优参数)\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-08-28\n'

def gradAscent(dataMatIn, classLabels):
    if False:
        print('Hello World!')
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    (m, n) = np.shape(dataMatrix)
    alpha = 0.01
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()
'\n函数说明:使用Python写的Logistic分类器做预测\n\nParameters:\n\t无\nReturns:\n\t无\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-09-05\n'

def colicTest():
    if False:
        while True:
            i = 10
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
            errorCount += 1
    errorRate = float(errorCount) / numTestVec * 100
    print('测试集错误率为: %.2f%%' % errorRate)
'\n函数说明:分类函数\n\nParameters:\n\tinX - 特征向量\n\tweights - 回归系数\nReturns:\n\t分类结果\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-09-05\n'

def classifyVector(inX, weights):
    if False:
        return 10
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
'\n函数说明:使用Sklearn构建Logistic回归分类器\n\nParameters:\n\t无\nReturns:\n\t无\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-09-05\n'

def colicSklearn():
    if False:
        for i in range(10):
            print('nop')
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    classifier = LogisticRegression(solver='sag', max_iter=5000).fit(trainingSet, trainingLabels)
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print('正确率:%f%%' % test_accurcy)
if __name__ == '__main__':
    colicSklearn()