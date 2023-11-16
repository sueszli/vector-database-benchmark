import numpy as np
import operator
import collections
'\n函数说明:创建数据集\n\nParameters:\n\t无\nReturns:\n\tgroup - 数据集\n\tlabels - 分类标签\nModify:\n\t2017-07-13\n'

def createDataSet():
    if False:
        i = 10
        return i + 15
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return (group, labels)
'\n函数说明:kNN算法,分类器\n\nParameters:\n\tinX - 用于分类的数据(测试集)\n\tdataSet - 用于训练的数据(训练集)\n\tlabes - 分类标签\n\tk - kNN算法参数,选择距离最小的k个点\nReturns:\n\tsortedClassCount[0][0] - 分类结果\n\nModify:\n\t2017-11-09 by Cugtyt \n\t\t* GitHub(https://github.com/Cugtyt) \n\t\t* Email(cugtyt@qq.com)\n\t\tUse list comprehension and Counter to simplify code\n\t2017-07-13\n'

def classify0(inx, dataset, labels, k):
    if False:
        i = 10
        return i + 15
    dist = np.sum((inx - dataset) ** 2, axis=1) ** 0.5
    k_labels = [labels[index] for index in dist.argsort()[0:k]]
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label
if __name__ == '__main__':
    (group, labels) = createDataSet()
    test = [101, 20]
    test_class = classify0(test, group, labels, 3)
    print(test_class)