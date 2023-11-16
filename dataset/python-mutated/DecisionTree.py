"""
Created on Oct 12, 2010
Update on 2017-05-18
Decision Tree Source Code for Machine Learning in Action Ch. 3
Author: Peter Harrington/片刻
GitHub: https://github.com/apachecn/AiLearning
"""
print(__doc__)
import operator
from math import log
import decisionTreePlot as dtPlot
from collections import Counter

def createDataSet():
    if False:
        return 10
    '\n    Desc:\n        创建数据集\n    Args:\n        无需传入参数\n    Returns:\n        返回数据集和对应的label标签\n    '
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return (dataSet, labels)

def calcShannonEnt(dataSet):
    if False:
        return 10
    '\n    Desc: \n        calculate Shannon entropy -- 计算给定数据集的香农熵\n    Args:\n        dataSet -- 数据集\n    Returns:\n        shannonEnt -- 返回 每一组 feature 下的某个分类下，香农熵的信息期望\n    '
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, index, value):
    if False:
        for i in range(10):
            print('nop')
    '\n    Desc: \n        划分数据集\n        splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)\n        就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中\n    Args:\n        dataSet  -- 数据集                 待划分的数据集\n        index -- 表示每一行的index列        划分数据集的特征\n        value -- 表示index列对应的value值   需要返回的特征的值。\n    Returns:\n        index 列为 value 的数据集【该数据集需要排除index列】\n    '
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]
            '\n            请百度查询一下:  extend和append的区别\n            list.append(object) 向列表中添加一个对象object\n            list.extend(sequence) 把一个序列seq的内容添加到列表中\n            1、使用append的时候，是将new_media看作一个对象，整体打包添加到music_media对象中。\n            2、使用extend的时候，是将new_media看作一个序列，将这个序列和music_media序列合并，并放在其后面。\n            result = []\n            result.extend([1,2,3])\n            print(result)\n            result.append([4,5,6])\n            print(result)\n            result.extend([7,8,9])\n            print(result)\n            结果: \n            [1, 2, 3]\n            [1, 2, 3, [4, 5, 6]]\n            [1, 2, 3, [4, 5, 6], 7, 8, 9]\n            '
            reducedFeatVec.extend(featVec[index + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    if False:
        i = 10
        return i + 15
    '\n    Desc:\n        选择切分数据集的最佳特征\n    Args:\n        dataSet -- 需要切分的数据集\n    Returns:\n        bestFeature -- 切分数据集的最优的特征列\n    '
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    (bestInfoGain, bestFeature) = (0.0, -1)
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    if False:
        while True:
            i = 10
    '\n    Desc:\n        选择出现次数最多的一个结果\n    Args:\n        classList label列的集合\n    Returns:\n        bestFeature 最优的特征列\n    '
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    if False:
        return 10
    '\n    Desc:\n        创建决策树\n    Args:\n        dataSet -- 要创建决策树的训练数据集\n        labels -- 训练数据集中特征对应的含义的labels，不是目标变量\n    Returns:\n        myTree -- 创建完成的决策树\n    '
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    if False:
        while True:
            i = 10
    '\n    Desc:\n        对新数据进行分类\n    Args:\n        inputTree  -- 已经训练好的决策树模型\n        featLabels -- Feature标签对应的名称，不是目标变量\n        testVec    -- 测试输入的数据\n    Returns:\n        classLabel -- 分类的结果值，需要映射label才能知道名称\n    '
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree, filename):
    if False:
        print('Hello World!')
    '\n    Desc:\n        将之前训练好的决策树模型存储起来，使用 pickle 模块\n    Args:\n        inputTree -- 以前训练好的决策树模型\n        filename -- 要存储的名称\n    Returns:\n        None\n    '
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)

def grabTree(filename):
    if False:
        for i in range(10):
            print('nop')
    '\n    Desc:\n        将之前存储的决策树模型使用 pickle 模块 还原出来\n    Args:\n        filename -- 之前存储决策树模型的文件名\n    Returns:\n        pickle.load(fr) -- 将之前存储的决策树模型还原出来\n    '
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

def fishTest():
    if False:
        while True:
            i = 10
    '\n    Desc:\n        对动物是否是鱼类分类的测试函数，并将结果使用 matplotlib 画出来\n    Args:\n        None\n    Returns:\n        None\n    '
    (myDat, labels) = createDataSet()
    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    print(classify(myTree, labels, [1, 1]))
    dtPlot.createPlot(myTree)

def ContactLensesTest():
    if False:
        for i in range(10):
            print('nop')
    '\n    Desc:\n        预测隐形眼镜的测试代码，并将结果画出来\n    Args:\n        none\n    Returns:\n        none\n    '
    fr = open('data/3.DecisionTree/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    dtPlot.createPlot(lensesTree)
if __name__ == '__main__':
    ContactLensesTest()