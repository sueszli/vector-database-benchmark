import matplotlib.pyplot as plt
import numpy as np
import random
'\nAuthor:\n\tJack Cui\nBlog:\n    http://blog.csdn.net/c406495762\nZhihu:\n    https://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-10-03\n'

class optStruct:
    """
	数据结构，维护所有需要操作的值
	Parameters：
		dataMatIn - 数据矩阵
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
	"""

    def __init__(self, dataMatIn, classLabels, C, toler):
        if False:
            print('Hello World!')
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))

def loadDataSet(fileName):
    if False:
        i = 10
        return i + 15
    '\n\t读取数据\n\tParameters:\n\t    fileName - 文件名\n\tReturns:\n\t    dataMat - 数据矩阵\n\t    labelMat - 数据标签\n\t'
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return (dataMat, labelMat)

def calcEk(oS, k):
    if False:
        while True:
            i = 10
    '\n\t计算误差\n\tParameters：\n\t\toS - 数据结构\n\t\tk - 标号为k的数据\n\tReturns:\n\t    Ek - 标号为k的数据误差\n\t'
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T) + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJrand(i, m):
    if False:
        for i in range(10):
            print('nop')
    '\n\t函数说明:随机选择alpha_j的索引值\n\n\tParameters:\n\t    i - alpha_i的索引值\n\t    m - alpha参数个数\n\tReturns:\n\t    j - alpha_j的索引值\n\t'
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def selectJ(i, oS, Ei):
    if False:
        return 10
    '\n\t内循环启发方式2\n\tParameters：\n\t\ti - 标号为i的数据的索引值\n\t\toS - 数据结构\n\t\tEi - 标号为i的数据误差\n\tReturns:\n\t    j, maxK - 标号为j或maxK的数据的索引值\n\t    Ej - 标号为j的数据误差\n\t'
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return (maxK, Ej)
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return (j, Ej)

def updateEk(oS, k):
    if False:
        i = 10
        return i + 15
    '\n\t计算Ek,并更新误差缓存\n\tParameters：\n\t\toS - 数据结构\n\t\tk - 标号为k的数据的索引值\n\tReturns:\n\t\t无\n\t'
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def clipAlpha(aj, H, L):
    if False:
        while True:
            i = 10
    '\n\t修剪alpha_j\n\tParameters:\n\t    aj - alpha_j的值\n\t    H - alpha上限\n\t    L - alpha下限\n\tReturns:\n\t    aj - 修剪后的alpah_j的值\n\t'
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def innerL(i, oS):
    if False:
        while True:
            i = 10
    '\n\t优化的SMO算法\n\tParameters：\n\t\ti - 标号为i的数据的索引值\n\t\toS - 数据结构\n\tReturns:\n\t\t1 - 有任意一对alpha值发生变化\n\t\t0 - 没有任意一对alpha值发生变化或变化太小\n\t'
    Ei = calcEk(oS, i)
    if oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C or (oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
        (j, Ej) = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L==H')
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print('eta>=0')
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 1e-05:
            print('alpha_j变化太小')
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if 0 < oS.alphas[i] and oS.C > oS.alphas[i]:
            oS.b = b1
        elif 0 < oS.alphas[j] and oS.C > oS.alphas[j]:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):
    if False:
        for i in range(10):
            print('nop')
    '\n\t完整的线性SMO算法\n\tParameters：\n\t\tdataMatIn - 数据矩阵\n\t\tclassLabels - 数据标签\n\t\tC - 松弛变量\n\t\ttoler - 容错率\n\t\tmaxIter - 最大迭代次数\n\tReturns:\n\t\toS.b - SMO算法计算的b\n\t\toS.alphas - SMO算法计算的alphas\n\t'
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print('全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d' % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d' % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print('迭代次数: %d' % iter)
    return (oS.b, oS.alphas)

def showClassifer(dataMat, classLabels, w, b):
    if False:
        return 10
    '\n\t分类结果可视化\n\tParameters:\n\t\tdataMat - 数据矩阵\n\t    w - 直线法向量\n\t    b - 直线解决\n\tReturns:\n\t    无\n\t'
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
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
        if alpha > 0:
            (x, y) = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()

def calcWs(alphas, dataArr, classLabels):
    if False:
        return 10
    '\n\t计算w\n\tParameters:\n\t\tdataArr - 数据矩阵\n\t    classLabels - 数据标签\n\t    alphas - alphas值\n\tReturns:\n\t    w - 计算得到的w\n\t'
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    (m, n) = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w
if __name__ == '__main__':
    (dataArr, classLabels) = loadDataSet('testSet.txt')
    (b, alphas) = smoP(dataArr, classLabels, 0.6, 0.001, 40)
    w = calcWs(alphas, dataArr, classLabels)
    showClassifer(dataArr, classLabels, w, b)