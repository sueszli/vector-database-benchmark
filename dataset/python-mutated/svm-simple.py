import matplotlib.pyplot as plt
import numpy as np
import random
'\n函数说明:读取数据\n\nParameters:\n    fileName - 文件名\nReturns:\n    dataMat - 数据矩阵\n    labelMat - 数据标签\nAuthor:\n    Jack Cui\nBlog:\n    http://blog.csdn.net/c406495762\nZhihu:\n    https://www.zhihu.com/people/Jack--Cui/\nModify:\n    2017-09-21\n'

def loadDataSet(fileName):
    if False:
        while True:
            i = 10
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return (dataMat, labelMat)
'\n函数说明:随机选择alpha\n\nParameters:\n    i - alpha_i的索引值\n    m - alpha参数个数\nReturns:\n    j - alpha_j的索引值\nAuthor:\n    Jack Cui\nBlog:\n    http://blog.csdn.net/c406495762\nZhihu:\n    https://www.zhihu.com/people/Jack--Cui/\nModify:\n    2017-09-21\n'

def selectJrand(i, m):
    if False:
        i = 10
        return i + 15
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j
'\n函数说明:修剪alpha\n\nParameters:\n    aj - alpha_j值\n    H - alpha上限\n    L - alpha下限\nReturns:\n    aj - alpah值\nAuthor:\n    Jack Cui\nBlog:\n    http://blog.csdn.net/c406495762\nZhihu:\n    https://www.zhihu.com/people/Jack--Cui/\nModify:\n    2017-09-21\n'

def clipAlpha(aj, H, L):
    if False:
        for i in range(10):
            print('nop')
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj
'\n函数说明:数据可视化\n\nParameters:\n    dataMat - 数据矩阵\n    labelMat - 数据标签\nReturns:\n    无\nAuthor:\n    Jack Cui\nBlog:\n    http://blog.csdn.net/c406495762\nZhihu:\n    https://www.zhihu.com/people/Jack--Cui/\nModify:\n    2017-09-21\n'

def showDataSet(dataMat, labelMat):
    if False:
        while True:
            i = 10
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()
'\n函数说明:简化版SMO算法\n\nParameters:\n    dataMatIn - 数据矩阵\n    classLabels - 数据标签\n    C - 松弛变量\n    toler - 容错率\n    maxIter - 最大迭代次数\nReturns:\n    无\nAuthor:\n    Jack Cui\nBlog:\n    http://blog.csdn.net/c406495762\nZhihu:\n    https://www.zhihu.com/people/Jack--Cui/\nModify:\n    2017-09-23\n'

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    if False:
        print('Hello World!')
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    (m, n) = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter_num = 0
    while iter_num < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if labelMat[i] * Ei < -toler and alphas[i] < C or (labelMat[i] * Ei > toler and alphas[i] > 0):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L==H')
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print('eta>=0')
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 1e-05:
                    print('alpha_j变化太小')
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if 0 < alphas[i] and C > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('第%d次迭代 样本:%d, alpha优化次数:%d' % (iter_num, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter_num += 1
        else:
            iter_num = 0
        print('迭代次数: %d' % iter_num)
    return (b, alphas)
'\n函数说明:分类结果可视化\n\nParameters:\n\tdataMat - 数据矩阵\n    w - 直线法向量\n    b - 直线解决\nReturns:\n    无\nAuthor:\n    Jack Cui\nBlog:\n    http://blog.csdn.net/c406495762\nZhihu:\n    https://www.zhihu.com/people/Jack--Cui/\nModify:\n    2017-09-23\n'

def showClassifer(dataMat, w, b):
    if False:
        i = 10
        return i + 15
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    (a1, a2) = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    (y1, y2) = ((-b - a1 * x1) / a2, (-b - a1 * x2) / a2)
    plt.plot([x1, x2], [y1, y2])
    for (i, alpha) in enumerate(alphas):
        if abs(alpha) > 0:
            (x, y) = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()
'\n函数说明:计算w\n\nParameters:\n\tdataMat - 数据矩阵\n    labelMat - 数据标签\n    alphas - alphas值\nReturns:\n    无\nAuthor:\n    Jack Cui\nBlog:\n    http://blog.csdn.net/c406495762\nZhihu:\n    https://www.zhihu.com/people/Jack--Cui/\nModify:\n    2017-09-23\n'

def get_w(dataMat, labelMat, alphas):
    if False:
        return 10
    (alphas, dataMat, labelMat) = (np.array(alphas), np.array(dataMat), np.array(labelMat))
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()
if __name__ == '__main__':
    (dataMat, labelMat) = loadDataSet('testSet.txt')
    (b, alphas) = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifer(dataMat, w, b)