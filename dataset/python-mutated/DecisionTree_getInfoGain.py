from math import log

def calcShannonEnt(dataSet):
    if False:
        print('Hello World!')
    'calcShannonEnt(calculate Shannon entropy 计算label分类标签的香农熵)\n\n    Args:\n        dataSet 数据集\n    Returns:\n        返回香农熵的计算值\n    Raises:\n\n    '
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

def splitDataSet(dataSet, axis, value):
    if False:
        return 10
    'splitDataSet(通过遍历dataSet数据集，求出axis对应的colnum列的值为value的行)\n\n    Args:\n        dataSet 数据集\n        axis 表示每一行的axis列\n        value 表示axis列对应的value值\n    Returns:\n        axis列为value的数据集【该数据集需要排除axis列】\n    Raises:\n\n    '
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            '\n            请百度查询一下:  extend和append的区别\n            '
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def getFeatureShannonEnt(dataSet, labels):
    if False:
        return 10
    'chooseBestFeatureToSplit(选择最好的特征)\n\n    Args:\n        dataSet 数据集\n    Returns:\n        bestFeature 最优的特征列\n    Raises:\n\n    '
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    (bestInfoGain, bestFeature, endEntropy) = (0.0, -1, 0.0)
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
            endEntropy = newEntropy
            bestInfoGain = infoGain
            bestFeature = i
    else:
        if numFeatures < 0:
            labels[bestFeature] = 'null'
    return (labels[bestFeature], baseEntropy, endEntropy, bestInfoGain)
if __name__ == '__main__':
    labels = ['no surfacing', 'flippers']
    dataSet1 = [['yes'], ['yes'], ['no'], ['no'], ['no']]
    dataSet2 = [['a', 1, 'yes'], ['a', 2, 'yes'], ['b', 3, 'no'], ['c', 4, 'no'], ['c', 5, 'no']]
    dataSet3 = [[1, 'yes'], [1, 'yes'], [1, 'no'], [3, 'no'], [3, 'no']]
    infoGain1 = getFeatureShannonEnt(dataSet1, labels)
    infoGain2 = getFeatureShannonEnt(dataSet2, labels)
    infoGain3 = getFeatureShannonEnt(dataSet3, labels)
    print('信息增益: \n\t%s, \n\t%s, \n\t%s' % (infoGain1, infoGain2, infoGain3))