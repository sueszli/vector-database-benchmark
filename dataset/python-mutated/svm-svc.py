import numpy as np
import operator
from os import listdir
from sklearn.svm import SVC
'\nAuthor:\n\tJack Cui\nBlog:\n    http://blog.csdn.net/c406495762\nZhihu:\n    https://www.zhihu.com/people/Jack--Cui/\nModify:\n\t2017-10-04\n'

def img2vector(filename):
    if False:
        for i in range(10):
            print('nop')
    '\n\t将32x32的二进制图像转换为1x1024向量。\n\tParameters:\n\t\tfilename - 文件名\n\tReturns:\n\t\treturnVect - 返回的二进制图像的1x1024向量\n\t'
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    if False:
        return 10
    '\n\t手写数字分类测试\n\tParameters:\n\t\t无\n\tReturns:\n\t\t无\n\t'
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    clf = SVC(C=200, kernel='rbf')
    clf.fit(trainingMat, hwLabels)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = clf.predict(vectorUnderTest)
        print('分类返回结果为%d\t真实结果为%d' % (classifierResult, classNumber))
        if classifierResult != classNumber:
            errorCount += 1.0
    print('总共错了%d个数据\n错误率为%f%%' % (errorCount, errorCount / mTest * 100))
if __name__ == '__main__':
    handwritingClassTest()