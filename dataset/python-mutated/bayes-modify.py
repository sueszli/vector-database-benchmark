import numpy as np
import random
import re
'\n函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表\n\nParameters:\n\tdataSet - 整理的样本数据集\nReturns:\n\tvocabSet - 返回不重复的词条列表，也就是词汇表\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-08-11\n'

def createVocabList(dataSet):
    if False:
        return 10
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
'\n函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0\n\nParameters:\n\tvocabList - createVocabList返回的列表\n\tinputSet - 切分的词条列表\nReturns:\n\treturnVec - 文档向量,词集模型\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-08-11\n'

def setOfWords2Vec(vocabList, inputSet):
    if False:
        while True:
            i = 10
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec
'\n函数说明:根据vocabList词汇表，构建词袋模型\n\nParameters:\n\tvocabList - createVocabList返回的列表\n\tinputSet - 切分的词条列表\nReturns:\n\treturnVec - 文档向量,词袋模型\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-08-14\n'

def bagOfWords2VecMN(vocabList, inputSet):
    if False:
        while True:
            i = 10
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
'\n函数说明:朴素贝叶斯分类器训练函数\n\nParameters:\n\ttrainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵\n\ttrainCategory - 训练类别标签向量，即loadDataSet返回的classVec\nReturns:\n\tp0Vect - 非侮辱类的条件概率数组\n\tp1Vect - 侮辱类的条件概率数组\n\tpAbusive - 文档属于侮辱类的概率\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-08-12\n'

def trainNB0(trainMatrix, trainCategory):
    if False:
        print('Hello World!')
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return (p0Vect, p1Vect, pAbusive)
'\n函数说明:朴素贝叶斯分类器分类函数\n\nParameters:\n\tvec2Classify - 待分类的词条数组\n\tp0Vec - 非侮辱类的条件概率数组\n\tp1Vec -侮辱类的条件概率数组\n\tpClass1 - 文档属于侮辱类的概率\nReturns:\n\t0 - 属于非侮辱类\n\t1 - 属于侮辱类\nAuthor:\n\tJack Cui\nBlog:\n\thttp://blog.csdn.net/c406495762\nModify:\n\t2017-08-12\n'

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    if False:
        i = 10
        return i + 15
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
'\n函数说明:接收一个大字符串并将其解析为字符串列表\n\nParameters:\n    无\nReturns:\n    无\nAuthor:\n    Jack Cui\nBlog:\n    http://blog.csdn.net/c406495762\nModify:\n    2017-08-14\n'

def textParse(bigString):
    if False:
        for i in range(10):
            print('nop')
    listOfTokens = re.split('\\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
'\n函数说明:测试朴素贝叶斯分类器\n\nParameters:\n    无\nReturns:\n    无\nAuthor:\n    Jack Cui\nBlog:\n    http://blog.csdn.net/c406495762\nModify:\n    2017-08-14\n'

def spamTest():
    if False:
        print('Hello World!')
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    (p0V, p1V, pSpam) = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print('分类错误的测试集：', docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))
if __name__ == '__main__':
    spamTest()