from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator
'\n函数说明:kNN算法,分类器\n\nParameters:\n\tinX - 用于分类的数据(测试集)\n\tdataSet - 用于训练的数据(训练集)\n\tlabes - 分类标签\n\tk - kNN算法参数,选择距离最小的k个点\nReturns:\n\tsortedClassCount[0][0] - 分类结果\n\nModify:\n\t2017-03-24\n'

def classify0(inX, dataSet, labels, k):
    if False:
        return 10
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]
'\n函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力\n\nParameters:\n\tfilename - 文件名\nReturns:\n\treturnMat - 特征矩阵\n\tclassLabelVector - 分类Label向量\n\nModify:\n\t2017-03-24\n'

def file2matrix(filename):
    if False:
        i = 10
        return i + 15
    fr = open(filename, 'r', encoding='utf-8')
    arrayOLines = fr.readlines()
    arrayOLines[0] = arrayOLines[0].lstrip('\ufeff')
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return (returnMat, classLabelVector)
'\n函数说明:可视化数据\n\nParameters:\n\tdatingDataMat - 特征矩阵\n\tdatingLabels - 分类Label\nReturns:\n\t无\nModify:\n\t2017-03-24\n'

def showdatas(datingDataMat, datingLabels):
    if False:
        print('Hello World!')
    font = FontProperties(fname='c:\\windows\\fonts\\simsunb.ttf', size=14)
    (fig, axs) = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=0.5)
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=0.5)
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=0.5)
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    plt.show()
'\n函数说明:对数据进行归一化\n\nParameters:\n\tdataSet - 特征矩阵\nReturns:\n\tnormDataSet - 归一化后的特征矩阵\n\tranges - 数据范围\n\tminVals - 数据最小值\n\nModify:\n\t2017-03-24\n'

def autoNorm(dataSet):
    if False:
        for i in range(10):
            print('nop')
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return (normDataSet, ranges, minVals)
'\n函数说明:分类器测试函数\n取百分之十的数据作为测试数据，检测分类器的正确性\n\nParameters:\n\t无\nReturns:\n\t无\n\nModify:\n\t2017-03-24\n'

def datingClassTest():
    if False:
        print('Hello World!')
    filename = 'datingTestSet.txt'
    (datingDataMat, datingLabels) = file2matrix(filename)
    hoRatio = 0.1
    (normMat, ranges, minVals) = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print('分类结果:%s\t真实类别:%d' % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print('错误率:%f%%' % (errorCount / float(numTestVecs) * 100))
'\n函数说明:通过输入一个人的三维特征,进行分类输出\n\nParameters:\n\t无\nReturns:\n\t无\n\nModify:\n\t2017-03-24\n'

def classifyPerson():
    if False:
        return 10
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    precentTats = float(input('玩视频游戏所耗时间百分比:'))
    ffMiles = float(input('每年获得的飞行常客里程数:'))
    iceCream = float(input('每周消费的冰激淋公升数:'))
    filename = 'datingTestSet.txt'
    (datingDataMat, datingLabels) = file2matrix(filename)
    (normMat, ranges, minVals) = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, precentTats, iceCream])
    norminArr = (inArr - minVals) / ranges
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    print('你可能%s这个人' % resultList[classifierResult - 1])
'\n函数说明:main函数\n\nParameters:\n\t无\nReturns:\n\t无\n\nModify:\n\t2017-03-24\n'
if __name__ == '__main__':
    datingClassTest()