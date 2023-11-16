from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random
'\n函数说明:加载数据\n\nParameters:\n\t无\nReturns:\n\tdataMat - 数据列表\n\tlabelMat - 标签列表\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-08-28\n'

def loadDataSet():
    if False:
        return 10
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return (dataMat, labelMat)
'\n函数说明:sigmoid函数\n\nParameters:\n\tinX - 数据\nReturns:\n\tsigmoid函数\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-08-28\n'

def sigmoid(inX):
    if False:
        print('Hello World!')
    return 1.0 / (1 + np.exp(-inX))
'\n函数说明:梯度上升算法\n\nParameters:\n\tdataMatIn - 数据集\n\tclassLabels - 数据标签\nReturns:\n\tweights.getA() - 求得的权重数组(最优参数)\n\tweights_array - 每次更新的回归系数\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-08-28\n'

def gradAscent(dataMatIn, classLabels):
    if False:
        i = 10
        return i + 15
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    (m, n) = np.shape(dataMatrix)
    alpha = 0.01
    maxCycles = 500
    weights = np.ones((n, 1))
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    return (weights.getA(), weights_array)
'\n函数说明:改进的随机梯度上升算法\n\nParameters:\n\tdataMatrix - 数据数组\n\tclassLabels - 数据标签\n\tnumIter - 迭代次数\nReturns:\n\tweights - 求得的回归系数数组(最优参数)\n\tweights_array - 每次更新的回归系数\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-08-31\n'

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    if False:
        return 10
    (m, n) = np.shape(dataMatrix)
    weights = np.ones(n)
    weights_array = np.array([])
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            weights_array = np.append(weights_array, weights, axis=0)
            del dataIndex[randIndex]
    weights_array = weights_array.reshape(numIter * m, n)
    return (weights, weights_array)
'\n函数说明:绘制数据集\n\nParameters:\n\tweights - 权重参数数组\nReturns:\n\t无\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-08-30\n'

def plotBestFit(weights):
    if False:
        return 10
    (dataMat, labelMat) = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=0.5)
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=0.5)
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
'\n函数说明:绘制回归系数与迭代次数的关系\n\nParameters:\n\tweights_array1 - 回归系数数组1\n\tweights_array2 - 回归系数数组2\nReturns:\n\t无\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-08-30\n'

def plotWeights(weights_array1, weights_array2):
    if False:
        return 10
    font = FontProperties(fname='c:\\windows\\fonts\\simsun.ttc', size=14)
    (fig, axs) = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    x1 = np.arange(0, len(weights_array1), 1)
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    x2 = np.arange(0, len(weights_array2), 1)
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    plt.show()
if __name__ == '__main__':
    (dataMat, labelMat) = loadDataSet()
    (weights1, weights_array1) = stocGradAscent1(np.array(dataMat), labelMat)
    (weights2, weights_array2) = gradAscent(dataMat, labelMat)
    plotWeights(weights_array1, weights_array2)