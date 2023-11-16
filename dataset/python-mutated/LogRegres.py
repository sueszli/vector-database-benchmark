import matplotlib.pyplot as plt
import numpy as np
'\n函数说明:梯度上升算法测试函数\n\n求函数f(x) = -x^2 + 4x的极大值\n\nParameters:\n\t无\nReturns:\n\t无\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-08-28\n'

def Gradient_Ascent_test():
    if False:
        i = 10
        return i + 15

    def f_prime(x_old):
        if False:
            i = 10
            return i + 15
        return -2 * x_old + 4
    x_old = -1
    x_new = 0
    alpha = 0.01
    presision = 1e-08
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
    print(x_new)
'\n函数说明:加载数据\n\nParameters:\n\t无\nReturns:\n\tdataMat - 数据列表\n\tlabelMat - 标签列表\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-08-28\n'

def loadDataSet():
    if False:
        i = 10
        return i + 15
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
'\n函数说明:梯度上升算法\n\nParameters:\n\tdataMatIn - 数据集\n\tclassLabels - 数据标签\nReturns:\n\tweights.getA() - 求得的权重数组(最优参数)\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-08-28\n'

def gradAscent(dataMatIn, classLabels):
    if False:
        return 10
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    (m, n) = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()
'\n函数说明:绘制数据集\n\nParameters:\n\t无\nReturns:\n\t无\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nZhihu:\n\thttps://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-08-30\n'

def plotDataSet():
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
    plt.title('DataSet')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
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
if __name__ == '__main__':
    (dataMat, labelMat) = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights)