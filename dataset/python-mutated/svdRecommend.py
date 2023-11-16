"""
Created on Mar 8, 2011
Update  on 2017-12-12
Author: Peter Harrington/山上有课树/片刻/marsjhao
GitHub: https://github.com/apachecn/AiLearning
"""
from numpy import linalg as la
from numpy import *

def loadExData3():
    if False:
        while True:
            i = 10
    return [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0], [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 8], [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0], [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0], [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0], [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 6]]

def loadExData2():
    if False:
        i = 10
        return i + 15
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5], [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3], [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0], [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0], [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0], [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0], [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1], [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4], [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2], [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0], [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def loadExData():
    if False:
        i = 10
        return i + 15
    '\n    # 推荐引擎示例矩阵\n    return[[4, 4, 0, 2, 2],\n           [4, 0, 0, 3, 3],\n           [4, 0, 0, 1, 1],\n           [1, 1, 1, 2, 0],\n           [2, 2, 2, 0, 0],\n           [1, 1, 1, 0, 0],\n           [5, 5, 5, 0, 0]]\n    '
    return [[0, -1.6, 0.6], [0, 1.2, 0.8], [0, 0, 0], [0, 0, 0]]

def ecludSim(inA, inB):
    if False:
        while True:
            i = 10
    return 1.0 / (1.0 + la.norm(inA - inB))

def pearsSim(inA, inB):
    if False:
        while True:
            i = 10
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]

def cosSim(inA, inB):
    if False:
        print('Hello World!')
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

def standEst(dataMat, user, simMeas, item):
    if False:
        for i in range(10):
            print('nop')
    'standEst(计算某用户未评分物品中，以对该物品和其他物品评分的用户的物品相似度，然后进行综合评分)\n    Args:\n        dataMat         训练数据集\n        user            用户编号\n        simMeas         相似度计算方法\n        item            未评分的物品编号\n    Returns:\n        ratSimTotal/simTotal     评分（0～5之间的值）\n    '
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print('the %d and %d similarity is : %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

def svdEst(dataMat, user, simMeas, item):
    if False:
        return 10
    'svdEst( )\n    Args:\n        dataMat         训练数据集\n        user            用户编号\n        simMeas         相似度计算方法\n        item            未评分的物品编号\n    Returns:\n        ratSimTotal/simTotal     评分（0～5之间的值）\n    '
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    (U, Sigma, VT) = la.svd(dataMat)
    Sig4 = mat(eye(4) * Sigma[:4])
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    if False:
        i = 10
        return i + 15
    'svdEst( )\n    Args:\n        dataMat         训练数据集\n        user            用户编号\n        simMeas         相似度计算方法\n        estMethod       使用的推荐算法\n    Returns:\n        返回最终 N 个推荐结果\n    '
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

def analyse_data(Sigma, loopNum=20):
    if False:
        for i in range(10):
            print('nop')
    'analyse_data(分析 Sigma 的长度取值)\n    Args:\n        Sigma         Sigma的值\n        loopNum       循环次数\n    '
    Sig2 = Sigma ** 2
    SigmaSum = sum(Sig2)
    for i in range(loopNum):
        SigmaI = sum(Sig2[:i + 1])
        '\n        根据自己的业务情况，就行处理，设置对应的 Singma 次数\n        通常保留矩阵 80% ～ 90% 的能量，就可以得到重要的特征并取出噪声。\n        '
        print('主成分: %s, 方差占比: %s%%' % (format(i + 1, '2.0f'), format(SigmaI / SigmaSum * 100, '4.2f')))

def imgLoadData(filename):
    if False:
        for i in range(10):
            print('nop')
    myl = []
    for line in open(filename).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    return myMat

def printMat(inMat, thresh=0.8):
    if False:
        i = 10
        return i + 15
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1)
            else:
                print(0)
        print('')

def imgCompress(numSV=3, thresh=0.8):
    if False:
        return 10
    'imgCompress( )\n    Args:\n        numSV       Sigma长度   \n        thresh      判断的阈值\n    '
    myMat = imgLoadData('data/14.SVD/0_5.txt')
    print('****original matrix****')
    printMat(myMat, thresh)
    (U, Sigma, VT) = la.svd(myMat)
    analyse_data(Sigma, 20)
    SigRecon = mat(eye(numSV) * Sigma[:numSV])
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print('****reconstructed matrix using %d singular values *****' % numSV)
    printMat(reconMat, thresh)
if __name__ == '__main__':
    '\n    # 计算欧氏距离\n    myMat = mat(loadExData())\n    # print(myMat)\n    print(ecludSim(myMat[:, 0], myMat[:, 4]))\n    print(ecludSim(myMat[:, 0], myMat[:, 0]))\n    # 计算余弦相似度\n    print(cosSim(myMat[:, 0], myMat[:, 4]))\n    print(cosSim(myMat[:, 0], myMat[:, 0]))\n    # 计算皮尔逊相关系数\n    print(pearsSim(myMat[:, 0], myMat[:, 4]))\n    print(pearsSim(myMat[:, 0], myMat[:, 0]))\n    '
    myMat = mat(loadExData3())
    print(recommend(myMat, 1, estMethod=svdEst))