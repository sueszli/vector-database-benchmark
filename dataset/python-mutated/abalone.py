from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
    if False:
        for i in range(10):
            print('nop')
    '\n\t函数说明:加载数据\n\tParameters:\n\t\tfileName - 文件名\n\tReturns:\n\t\txArr - x数据集\n\t\tyArr - y数据集\n\tWebsite:\n\t\thttp://www.cuijiahua.com/\n\tModify:\n\t\t2017-11-19\n\t'
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

def lwlr(testPoint, xArr, yArr, k=1.0):
    if False:
        i = 10
        return i + 15
    '\n\t函数说明:使用局部加权线性回归计算回归系数w\n\tParameters:\n\t\ttestPoint - 测试样本点\n\t\txArr - x数据集\n\t\tyArr - y数据集\n\t\tk - 高斯核的k,自定义参数\n\tReturns:\n\t\tws - 回归系数\n\tWebsite:\n\t\thttp://www.cuijiahua.com/\n\tModify:\n\t\t2017-11-19\n\t'
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
        return 10
    '\n\t函数说明:局部加权线性回归测试\n\tParameters:\n\t\ttestArr - 测试数据集,测试集\n\t\txArr - x数据集,训练集\n\t\tyArr - y数据集,训练集\n\t\tk - 高斯核的k,自定义参数\n\tReturns:\n\t\tws - 回归系数\n\tWebsite:\n\t\thttp://www.cuijiahua.com/\n\tModify:\n\t\t2017-11-19\n\t'
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def standRegres(xArr, yArr):
    if False:
        i = 10
        return i + 15
    '\n\t函数说明:计算回归系数w\n\tParameters:\n\t\txArr - x数据集\n\t\tyArr - y数据集\n\tReturns:\n\t\tws - 回归系数\n\tWebsite:\n\t\thttp://www.cuijiahua.com/\n\tModify:\n\t\t2017-11-19\n\t'
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print('矩阵为奇异矩阵,不能求逆')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def rssError(yArr, yHatArr):
    if False:
        return 10
    '\n\t误差大小评价函数\n\tParameters:\n\t\tyArr - 真实数据\n\t\tyHatArr - 预测数据\n\tReturns:\n\t\t误差大小\n\t'
    return ((yArr - yHatArr) ** 2).sum()
if __name__ == '__main__':
    (abX, abY) = loadDataSet('abalone.txt')
    print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:', rssError(abY[0:99], yHat01.T))
    print('k=1  时,误差大小为:', rssError(abY[0:99], yHat1.T))
    print('k=10 时,误差大小为:', rssError(abY[0:99], yHat10.T))
    print('')
    print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:', rssError(abY[100:199], yHat01.T))
    print('k=1  时,误差大小为:', rssError(abY[100:199], yHat1.T))
    print('k=10 时,误差大小为:', rssError(abY[100:199], yHat10.T))
    print('')
    print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
    print('k=1时,误差大小为:', rssError(abY[100:199], yHat1.T))
    ws = standRegres(abX[0:99], abY[0:99])
    yHat = np.mat(abX[100:199]) * ws
    print('简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A))