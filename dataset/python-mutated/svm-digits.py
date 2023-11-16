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
		kTup - 包含核函数信息的元组,第一个参数存放核函数类别，第二个参数存放必要的核函数需要用到的参数
	"""

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
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
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

def kernelTrans(X, A, kTup):
    if False:
        return 10
    '\n\t通过核函数将数据转换更高维的空间\n\tParameters：\n\t\tX - 数据矩阵\n\t\tA - 单个数据的向量\n\t\tkTup - 包含核函数信息的元组\n\tReturns:\n\t    K - 计算的核K\n\t'
    (m, n) = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('核函数无法识别')
    return K

def loadDataSet(fileName):
    if False:
        for i in range(10):
            print('nop')
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
        return 10
    '\n\t计算误差\n\tParameters：\n\t\toS - 数据结构\n\t\tk - 标号为k的数据\n\tReturns:\n\t    Ek - 标号为k的数据误差\n\t'
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJrand(i, m):
    if False:
        while True:
            i = 10
    '\n\t函数说明:随机选择alpha_j的索引值\n\n\tParameters:\n\t    i - alpha_i的索引值\n\t    m - alpha参数个数\n\tReturns:\n\t    j - alpha_j的索引值\n\t'
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def selectJ(i, oS, Ei):
    if False:
        print('Hello World!')
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
        print('Hello World!')
    '\n\t计算Ek,并更新误差缓存\n\tParameters：\n\t\toS - 数据结构\n\t\tk - 标号为k的数据的索引值\n\tReturns:\n\t\t无\n\t'
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def clipAlpha(aj, H, L):
    if False:
        i = 10
        return i + 15
    '\n\t修剪alpha_j\n\tParameters:\n\t    aj - alpha_j的值\n\t    H - alpha上限\n\t    L - alpha下限\n\tReturns:\n\t    aj - 修剪后的alpah_j的值\n\t'
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def innerL(i, oS):
    if False:
        print('Hello World!')
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
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
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
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if 0 < oS.alphas[i] and oS.C > oS.alphas[i]:
            oS.b = b1
        elif 0 < oS.alphas[j] and oS.C > oS.alphas[j]:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    if False:
        for i in range(10):
            print('nop')
    '\n\t完整的线性SMO算法\n\tParameters：\n\t\tdataMatIn - 数据矩阵\n\t\tclassLabels - 数据标签\n\t\tC - 松弛变量\n\t\ttoler - 容错率\n\t\tmaxIter - 最大迭代次数\n\t\tkTup - 包含核函数信息的元组\n\tReturns:\n\t\toS.b - SMO算法计算的b\n\t\toS.alphas - SMO算法计算的alphas\n\t'
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
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

def img2vector(filename):
    if False:
        return 10
    '\n\t将32x32的二进制图像转换为1x1024向量。\n\tParameters:\n\t\tfilename - 文件名\n\tReturns:\n\t\treturnVect - 返回的二进制图像的1x1024向量\n\t'
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    if False:
        print('Hello World!')
    '\n\t加载图片\n\tParameters:\n\t\tdirName - 文件夹的名字\n\tReturns:\n\t    trainingMat - 数据矩阵\n\t    hwLabels - 数据标签\n\t'
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return (trainingMat, hwLabels)

def testDigits(kTup=('rbf', 10)):
    if False:
        i = 10
        return i + 15
    '\n\t测试函数\n\tParameters:\n\t\tkTup - 包含核函数信息的元组\n\tReturns:\n\t    无\n\t'
    (dataArr, labelArr) = loadImages('trainingDigits')
    (b, alphas) = smoP(dataArr, labelArr, 200, 0.0001, 10, kTup)
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print('支持向量个数:%d' % np.shape(sVs)[0])
    (m, n) = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('训练集错误率: %.2f%%' % (float(errorCount) / m))
    (dataArr, labelArr) = loadImages('testDigits')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    (m, n) = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('测试集错误率: %.2f%%' % (float(errorCount) / m))
if __name__ == '__main__':
    testDigits()