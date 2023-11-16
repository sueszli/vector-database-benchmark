from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
    if False:
        for i in range(10):
            print('nop')
    '\n\t函数说明:加载数据\n\tParameters:\n\t\tfileName - 文件名\n\tReturns:\n\t\txArr - x数据集\n\t\tyArr - y数据集\n\tWebsite:\n\t\thttp://www.cuijiahua.com/\n\tModify:\n\t\t2017-11-12\n\t'
    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = []
    yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return (xArr, yArr)

def standRegres(xArr, yArr):
    if False:
        return 10
    '\n\t函数说明:计算回归系数w\n\tParameters:\n\t\txArr - x数据集\n\t\tyArr - y数据集\n\tReturns:\n\t\tws - 回归系数\n\tWebsite:\n\t\thttp://www.cuijiahua.com/\n\tModify:\n\t\t2017-11-12\n\t'
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print('矩阵为奇异矩阵,不能求逆')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def plotDataSet():
    if False:
        i = 10
        return i + 15
    '\n\t函数说明:绘制数据集\n\tParameters:\n\t\t无\n\tReturns:\n\t\t无\n\tWebsite:\n\t\thttp://www.cuijiahua.com/\n\tModify:\n\t\t2017-11-12\n\t'
    (xArr, yArr) = loadDataSet('ex0.txt')
    n = len(xArr)
    xcord = []
    ycord = []
    for i in range(n):
        xcord.append(xArr[i][1])
        ycord.append(yArr[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def plotRegression():
    if False:
        print('Hello World!')
    '\n\t函数说明:绘制回归曲线和数据点\n\tParameters:\n\t\t无\n\tReturns:\n\t\t无\n\tWebsite:\n\t\thttp://www.cuijiahua.com/\n\tModify:\n\t\t2017-11-12\n\t'
    (xArr, yArr) = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xCopy[:, 1], yHat, c='red')
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def plotlwlrRegression():
    if False:
        i = 10
        return i + 15
    '\n\t函数说明:绘制多条局部加权回归曲线\n\tParameters:\n\t\t无\n\tReturns:\n\t\t无\n\tWebsite:\n\t\thttp://www.cuijiahua.com/\n\tModify:\n\t\t2017-11-15\n\t'
    font = FontProperties(fname='c:\\windows\\fonts\\simsun.ttc', size=14)
    (xArr, yArr) = loadDataSet('ex0.txt')
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    (fig, axs) = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(10, 8))
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c='red')
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c='red')
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c='red')
    axs[0].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=0.5)
    axs[1].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=0.5)
    axs[2].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=0.5)
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0', FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01', FontProperties=font)
    axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003', FontProperties=font)
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()

def lwlr(testPoint, xArr, yArr, k=1.0):
    if False:
        while True:
            i = 10
    '\n\t函数说明:使用局部加权线性回归计算回归系数w\n\tParameters:\n\t\ttestPoint - 测试样本点\n\t\txArr - x数据集\n\t\tyArr - y数据集\n\t\tk - 高斯核的k,自定义参数\n\tReturns:\n\t\tws - 回归系数\n\tWebsite:\n\t\thttp://www.cuijiahua.com/\n\tModify:\n\t\t2017-11-15\n\t'
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('矩阵为奇异矩阵,不能求逆')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    if False:
        print('Hello World!')
    '\n\t函数说明:局部加权线性回归测试\n\tParameters:\n\t\ttestArr - 测试数据集\n\t\txArr - x数据集\n\t\tyArr - y数据集\n\t\tk - 高斯核的k,自定义参数\n\tReturns:\n\t\tws - 回归系数\n\tWebsite:\n\t\thttp://www.cuijiahua.com/\n\tModify:\n\t\t2017-11-15\n\t'
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat
if __name__ == '__main__':
    plotlwlrRegression()