"""
Created on Nov 4, 2010
Update on 2017-05-18
Chapter 5 source file for Machine Learing in Action
Author: Peter/geekidentity/片刻
GitHub: https://github.com/apachecn/AiLearning
"""
from numpy import *
import matplotlib.pyplot as plt

class optStruct:
    """
    建立的数据结构来保存所有的重要值
    """

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        if False:
            while True:
                i = 10
        '\n        Args:\n            dataMatIn    数据集\n            classLabels  类别标签\n            C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。\n                控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。\n                可以通过调节该参数达到不同的结果。\n            toler   容错率\n            kTup    包含核函数信息的元组\n        '
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i], kTup)

def kernelTrans(X, A, kTup):
    if False:
        for i in range(10):
            print('nop')
    '\n    核转换函数\n    Args:\n        X     dataMatIn数据集\n        A     dataMatIn数据集的第i行的数据\n        kTup  核函数的信息\n\n    Returns:\n\n    '
    (m, n) = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

def loadDataSet(fileName):
    if False:
        i = 10
        return i + 15
    'loadDataSet（对文件进行逐行解析，从而得到第行的类标签和整个数据矩阵）\n\n    Args:\n        fileName 文件名\n    Returns:\n        dataMat  数据矩阵\n        labelMat 类标签\n    '
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
    'calcEk（求 Ek误差: 预测值-真实值的差）\n\n    该过程在完整版的SMO算法中陪出现次数较多，因此将其单独作为一个方法\n    Args:\n        oS  optStruct对象\n        k   具体的某一行\n\n    Returns:\n        Ek  预测结果与真实结果比对，计算误差Ek\n    '
    fXk = multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJrand(i, m):
    if False:
        while True:
            i = 10
    '\n    随机选择一个整数\n    Args:\n        i  第一个alpha的下标\n        m  所有alpha的数目\n    Returns:\n        j  返回一个不为i的随机数，在0~m之间的整数值\n    '
    j = i
    while j == i:
        j = random.randint(0, m - 1)
    return j

def selectJ(i, oS, Ei):
    if False:
        while True:
            i = 10
    'selectJ（返回最优的j和Ej）\n\n    内循环的启发式方法。\n    选择第二个(内循环)alpha的alpha值\n    这里的目标是选择合适的第二个alpha值以保证每次优化中采用最大步长。\n    该函数的误差与第一个alpha值Ei和下标i有关。\n    Args:\n        i   具体的第i一行\n        oS  optStruct对象\n        Ei  预测结果与真实结果比对，计算误差Ei\n\n    Returns:\n        j  随机选出的第j一行\n        Ej 预测结果与真实结果比对，计算误差Ej\n    '
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
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
        for i in range(10):
            print('nop')
    'updateEk（计算误差值并存入缓存中。）\n\n    在对alpha值进行优化之后会用到这个值。\n    Args:\n        oS  optStruct对象\n        k   某一列的行号\n    '
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def clipAlpha(aj, H, L):
    if False:
        while True:
            i = 10
    'clipAlpha(调整aj的值，使aj处于 L<=aj<=H)\n    Args:\n        aj  目标值\n        H   最大值\n        L   最小值\n    Returns:\n        aj  目标值\n    '
    aj = min(aj, H)
    aj = max(L, aj)
    return aj

def innerL(i, oS):
    if False:
        print('Hello World!')
    'innerL\n    内循环代码\n    Args:\n        i   具体的某一行\n        oS  optStruct对象\n\n    Returns:\n        0   找不到最优的值\n        1   找到了最优的值，并且oS.Cache到缓存中\n    '
    Ei = calcEk(oS, i)
    '\n    # 检验训练样本(xi, yi)是否满足KKT条件\n    yi*f(i) >= 1 and alpha = 0 (outside the boundary)\n    yi*f(i) == 1 and 0<alpha< C (on the boundary)\n    yi*f(i) <= 1 and alpha = C (between the boundary)\n    '
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
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print('eta>=0')
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 1e-05:
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
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    if False:
        return 10
    '\n    完整SMO算法外循环，与smoSimple有些类似，但这里的循环退出条件更多一些\n    Args:\n        dataMatIn    数据集\n        classLabels  类别标签\n        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。\n            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。\n            可以通过调节该参数达到不同的结果。\n        toler   容错率\n        maxIter 退出前最大的循环次数\n        kTup    包含核函数信息的元组\n    Returns:\n        b       模型的常量值\n        alphas  拉格朗日乘子\n    '
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print('iteration number: %d' % iter)
    return (oS.b, oS.alphas)

def calcWs(alphas, dataArr, classLabels):
    if False:
        while True:
            i = 10
    '\n    基于alpha计算w值\n    Args:\n        alphas        拉格朗日乘子\n        dataArr       feature数据集\n        classLabels   目标变量数据集\n\n    Returns:\n        wc  回归系数\n    '
    X = mat(dataArr)
    labelMat = mat(classLabels).T
    (m, n) = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i].T)
    return w

def testRbf(k1=1.3):
    if False:
        i = 10
        return i + 15
    (dataArr, labelArr) = loadDataSet('data/6.SVM/testSetRBF.txt')
    (b, alphas) = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print('there are %d Support Vectors' % shape(sVs)[0])
    (m, n) = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('the training error rate is: %f' % (float(errorCount) / m))
    (dataArr, labelArr) = loadDataSet('data/6.SVM/testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    (m, n) = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('the test error rate is: %f' % (float(errorCount) / m))

def img2vector(filename):
    if False:
        while True:
            i = 10
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    if False:
        print('Hello World!')
    from os import listdir
    hwLabels = []
    print(dirName)
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
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
        print('Hello World!')
    (dataArr, labelArr) = loadImages('data/6.SVM/trainingDigits')
    (b, alphas) = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    (m, n) = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('the training error rate is: %f' % (float(errorCount) / m))
    (dataArr, labelArr) = loadImages('data/6.SVM/testDigits')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    (m, n) = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('the test error rate is: %f' % (float(errorCount) / m))

def plotfig_SVM(xArr, yArr, ws, b, alphas):
    if False:
        while True:
            i = 10
    '\n    参考地址: \n       http://blog.csdn.net/maoersong/article/details/24315633\n       http://www.cnblogs.com/JustForCS/p/5283489.html\n       http://blog.csdn.net/kkxgx/article/details/6951959\n    '
    xMat = mat(xArr)
    yMat = mat(yArr)
    b = array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])
    x = arange(-1.0, 10.0, 0.1)
    y = (-b - ws[0, 0] * x) / ws[1, 0]
    ax.plot(x, y)
    for i in range(shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()
if __name__ == '__main__':
    testDigits(('rbf', 10))