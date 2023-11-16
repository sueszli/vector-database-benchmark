import numpy as np
import operator
'\n函数说明:创建数据集\n\nParameters:\n\t无\nReturns:\n\tgroup - 数据集\n\tlabels - 分类标签\nModify:\n\t2017-07-13\n'

def createDataSet():
    if False:
        for i in range(10):
            print('nop')
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return (group, labels)
'\n函数说明:kNN算法,分类器\n\nParameters:\n\tinX - 用于分类的数据(测试集)\n\tdataSet - 用于训练的数据(训练集)\n\tlabes - 分类标签\n\tk - kNN算法参数,选择距离最小的k个点\nReturns:\n\tsortedClassCount[0][0] - 分类结果\n\nModify:\n\t2017-07-13\n'

def classify0(inX, dataSet, labels, k):
    if False:
        while True:
            i = 10
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
    return sortedClassCount[0][0]
if __name__ == '__main__':
    (group, labels) = createDataSet()
    test = [101, 20]
    test_class = classify0(test, group, labels, 3)
    print(test_class)