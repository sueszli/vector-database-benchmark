import numpy as np
import operator
from os import listdir
'\n函数说明:kNN算法,分类器\n\nParameters:\n\tinX - 用于分类的数据(测试集)\n\tdataSet - 用于训练的数据(训练集)\n\tlabes - 分类标签\n\tk - kNN算法参数,选择距离最小的k个点\nReturns:\n\tsortedClassCount[0][0] - 分类结果\n\nModify:\n\t2017-03-25\n'

def classify0(inX, dataSet, labels, k):
    if False:
        print('Hello World!')
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
'\n函数说明:将32x32的二进制图像转换为1x1024向量。\n\nParameters:\n\tfilename - 文件名\nReturns:\n\treturnVect - 返回的二进制图像的1x1024向量\n\nModify:\n\t2017-03-25\n'

def img2vector(filename):
    if False:
        print('Hello World!')
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect
'\n函数说明:手写数字分类测试\n\nParameters:\n\t无\nReturns:\n\t无\n\nModify:\n\t2017-03-25\n'

def handwritingClassTest():
    if False:
        print('Hello World!')
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('分类返回结果为%d\t真实结果为%d' % (classifierResult, classNumber))
        if classifierResult != classNumber:
            errorCount += 1.0
    print('总共错了%d个数据\n错误率为%f%%' % (errorCount, errorCount / mTest))
'\n函数说明:main函数\n\nParameters:\n\t无\nReturns:\n\t无\n\nModify:\n\t2017-03-25\n'
if __name__ == '__main__':
    handwritingClassTest()