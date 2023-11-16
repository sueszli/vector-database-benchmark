from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle
'\n函数说明:计算给定数据集的经验熵(香农熵)\n\nParameters:\n\tdataSet - 数据集\nReturns:\n\tshannonEnt - 经验熵(香农熵)\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-24\n'

def calcShannonEnt(dataSet):
    if False:
        i = 10
        return i + 15
    numEntires = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
'\n函数说明:创建测试数据集\n\nParameters:\n\t无\nReturns:\n\tdataSet - 数据集\n\tlabels - 特征标签\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-20\n'

def createDataSet():
    if False:
        for i in range(10):
            print('nop')
    dataSet = [[0, 0, 0, 0, 'no'], [0, 0, 0, 1, 'no'], [0, 1, 0, 1, 'yes'], [0, 1, 1, 0, 'yes'], [0, 0, 0, 0, 'no'], [1, 0, 0, 0, 'no'], [1, 0, 0, 1, 'no'], [1, 1, 1, 1, 'yes'], [1, 0, 1, 2, 'yes'], [1, 0, 1, 2, 'yes'], [2, 0, 1, 2, 'yes'], [2, 0, 1, 1, 'yes'], [2, 1, 0, 1, 'yes'], [2, 1, 0, 2, 'yes'], [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return (dataSet, labels)
'\n函数说明:按照给定特征划分数据集\n\nParameters:\n\tdataSet - 待划分的数据集\n\taxis - 划分数据集的特征\n\tvalue - 需要返回的特征的值\nReturns:\n\t无\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-24\n'

def splitDataSet(dataSet, axis, value):
    if False:
        for i in range(10):
            print('nop')
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
'\n函数说明:选择最优特征\n\nParameters:\n\tdataSet - 数据集\nReturns:\n\tbestFeature - 信息增益最大的(最优)特征的索引值\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-20\n'

def chooseBestFeatureToSplit(dataSet):
    if False:
        i = 10
        return i + 15
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
'\n函数说明:统计classList中出现此处最多的元素(类标签)\n\nParameters:\n\tclassList - 类标签列表\nReturns:\n\tsortedClassCount[0][0] - 出现此处最多的元素(类标签)\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-24\n'

def majorityCnt(classList):
    if False:
        i = 10
        return i + 15
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
'\n函数说明:创建决策树\n\nParameters:\n\tdataSet - 训练数据集\n\tlabels - 分类属性标签\n\tfeatLabels - 存储选择的最优特征标签\nReturns:\n\tmyTree - 决策树\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-25\n'

def createTree(dataSet, labels, featLabels):
    if False:
        while True:
            i = 10
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
    return myTree
'\n函数说明:获取决策树叶子结点的数目\n\nParameters:\n\tmyTree - 决策树\nReturns:\n\tnumLeafs - 决策树的叶子结点的数目\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-24\n'

def getNumLeafs(myTree):
    if False:
        print('Hello World!')
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs
'\n函数说明:获取决策树的层数\n\nParameters:\n\tmyTree - 决策树\nReturns:\n\tmaxDepth - 决策树的层数\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-24\n'

def getTreeDepth(myTree):
    if False:
        while True:
            i = 10
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth
'\n函数说明:绘制结点\n\nParameters:\n\tnodeTxt - 结点名\n\tcenterPt - 文本位置\n\tparentPt - 标注的箭头位置\n\tnodeType - 结点格式\nReturns:\n\t无\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-24\n'

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    if False:
        while True:
            i = 10
    arrow_args = dict(arrowstyle='<-')
    font = FontProperties(fname='c:\\windows\\fonts\\simsunb.ttf', size=14)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va='center', ha='center', bbox=nodeType, arrowprops=arrow_args, FontProperties=font)
'\n函数说明:标注有向边属性值\n\nParameters:\n\tcntrPt、parentPt - 用于计算标注位置\n\ttxtString - 标注的内容\nReturns:\n\t无\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-24\n'

def plotMidText(cntrPt, parentPt, txtString):
    if False:
        return 10
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va='center', ha='center', rotation=30)
'\n函数说明:绘制决策树\n\nParameters:\n\tmyTree - 决策树(字典)\n\tparentPt - 标注的内容\n\tnodeTxt - 结点名\nReturns:\n\t无\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-24\n'

def plotTree(myTree, parentPt, nodeTxt):
    if False:
        i = 10
        return i + 15
    decisionNode = dict(boxstyle='sawtooth', fc='0.8')
    leafNode = dict(boxstyle='round4', fc='0.8')
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
'\n函数说明:创建绘制面板\n\nParameters:\n\tinTree - 决策树(字典)\nReturns:\n\t无\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-24\n'

def createPlot(inTree):
    if False:
        print('Hello World!')
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
'\n函数说明:使用决策树分类\n\nParameters:\n\tinputTree - 已经生成的决策树\n\tfeatLabels - 存储选择的最优特征标签\n\ttestVec - 测试数据列表，顺序对应最优特征标签\nReturns:\n\tclassLabel - 分类结果\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-25\n'

def classify(inputTree, featLabels, testVec):
    if False:
        print('Hello World!')
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
'\n函数说明:存储决策树\n\nParameters:\n\tinputTree - 已经生成的决策树\n\tfilename - 决策树的存储文件名\nReturns:\n\t无\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-25\n'

def storeTree(inputTree, filename):
    if False:
        return 10
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
'\n函数说明:读取决策树\n\nParameters:\n\tfilename - 决策树的存储文件名\nReturns:\n\tpickle.load(fr) - 决策树字典\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-07-25\n'

def grabTree(filename):
    if False:
        return 10
    fr = open(filename, 'rb')
    return pickle.load(fr)
if __name__ == '__main__':
    (dataSet, labels) = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    createPlot(myTree)
    testVec = [0, 1]
    result = classify(myTree, featLabels, testVec)
    if result == 'yes':
        print('放贷')
    if result == 'no':
        print('不放贷')