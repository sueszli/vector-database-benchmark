"""
Created on 2017-06-28
Updated on 2017-06-28
SVM: 最大边距分离超平面
Author: 片刻
GitHub: https://github.com/apachecn/AiLearning
sklearn-SVM译文链接: http://cwiki.apachecn.org/pages/viewpage.action?pageId=10031359
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
print(__doc__)
np.random.seed(0)

def loadDataSet(fileName):
    if False:
        i = 10
        return i + 15
    '\n    对文件进行逐行解析，从而得到第行的类标签和整个数据矩阵\n    Args:\n        fileName 文件名\n    Returns:\n        dataMat  数据矩阵\n        labelMat 类标签\n    '
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return (dataMat, labelMat)
(X, Y) = loadDataSet('data/6.SVM/testSet.txt')
X = np.mat(X)
print('X=', X)
print('Y=', Y)
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2, 10)
yy = a * xx - clf.intercept_[0] / w[1]
print('yy=', yy)
print('support_vectors_=', clf.support_vectors_)
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
plt.scatter(X[:, 0].flat, X[:, 1].flat, c=Y, cmap=plt.cm.Paired)
plt.axis('tight')
plt.show()