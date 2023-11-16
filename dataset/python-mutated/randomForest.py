"""
Created 2017-04-25
Update  on 2017-05-18
Random Forest Algorithm on Sonar Dataset
Author: Flying_sfeng/片刻
GitHub: https://github.com/apachecn/AiLearning
---
源代码网址: http://www.tuicool.com/articles/iiUfeim
Flying_sfeng博客地址: http://blog.csdn.net/flying_sfeng/article/details/64133822
在此表示感谢你的代码和注解， 我重新也完善了个人注解
"""
from random import seed, randrange, random

def loadDataSet(filename):
    if False:
        i = 10
        return i + 15
    dataset = []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            if not line:
                continue
            lineArr = []
            for featrue in line.split(','):
                str_f = featrue.strip()
                if str_f.isdigit():
                    lineArr.append(float(str_f))
                else:
                    lineArr.append(str_f)
            dataset.append(lineArr)
    return dataset

def cross_validation_split(dataset, n_folds):
    if False:
        i = 10
        return i + 15
    'cross_validation_split(将数据集进行抽重抽样 n_folds 份，数据可以重复重复抽取，每一次list的元素是无重复的)\n\n    Args:\n        dataset     原始数据集\n        n_folds     数据集dataset分成n_flods份\n    Returns:\n        dataset_split    list集合，存放的是: 将数据集进行抽重抽样 n_folds 份，数据可以重复重复抽取，每一次list的元素是无重复的\n    '
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = len(dataset) / n_folds
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy[index])
        dataset_split.append(fold)
    return dataset_split

def test_split(index, value, dataset):
    if False:
        print('Hello World!')
    (left, right) = (list(), list())
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return (left, right)

def gini_index(groups, class_values):
    if False:
        while True:
            i = 10
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += proportion * (1.0 - proportion)
    return gini

def get_split(dataset, n_features):
    if False:
        i = 10
        return i + 15
    class_values = list(set((row[-1] for row in dataset)))
    (b_index, b_value, b_score, b_groups) = (999, 999, 999, None)
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                (b_index, b_value, b_score, b_groups) = (index, row[index], gini, groups)
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def to_terminal(group):
    if False:
        return 10
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, n_features, depth):
    if False:
        print('Hello World!')
    (left, right) = node['groups']
    del node['groups']
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        (node['left'], node['right']) = (to_terminal(left), to_terminal(right))
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)

def build_tree(train, max_depth, min_size, n_features):
    if False:
        for i in range(10):
            print('nop')
    'build_tree(创建一个决策树)\n\n    Args:\n        train           训练数据集\n        max_depth       决策树深度不能太深，不然容易导致过拟合\n        min_size        叶子节点的大小\n        n_features      选取的特征的个数\n    Returns:\n        root            返回决策树\n    '
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root

def predict(node, row):
    if False:
        for i in range(10):
            print('nop')
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    elif isinstance(node['right'], dict):
        return predict(node['right'], row)
    else:
        return node['right']

def bagging_predict(trees, row):
    if False:
        return 10
    'bagging_predict(bagging预测)\n\n    Args:\n        trees           决策树的集合\n        row             测试数据集的每一行数据\n    Returns:\n        返回随机森林中，决策树结果出现次数做大的\n    '
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def subsample(dataset, ratio):
    if False:
        while True:
            i = 10
    'random_forest(评估算法性能，返回模型得分)\n\n    Args:\n        dataset         训练数据集\n        ratio           训练数据集的样本比例\n    Returns:\n        sample          随机抽样的训练样本\n    '
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    if False:
        for i in range(10):
            print('nop')
    'random_forest(评估算法性能，返回模型得分)\n\n    Args:\n        train           训练数据集\n        test            测试数据集\n        max_depth       决策树深度不能太深，不然容易导致过拟合\n        min_size        叶子节点的大小\n        sample_size     训练数据集的样本比例\n        n_trees         决策树的个数\n        n_features      选取的特征的个数\n    Returns:\n        predictions     每一行的预测结果，bagging 预测最后的分类结果\n    '
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions

def accuracy_metric(actual, predicted):
    if False:
        for i in range(10):
            print('nop')
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    if False:
        i = 10
        return i + 15
    'evaluate_algorithm(评估算法性能，返回模型得分)\n\n    Args:\n        dataset     原始数据集\n        algorithm   使用的算法\n        n_folds     数据的份数\n        *args       其他的参数\n    Returns:\n        scores      模型得分\n    '
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        "\n        In [20]: l1=[[1, 2, 'a'], [11, 22, 'b']]\n        In [21]: l2=[[3, 4, 'c'], [33, 44, 'd']]\n        In [22]: l=[]\n        In [23]: l.append(l1)\n        In [24]: l.append(l2)\n        In [25]: l\n        Out[25]: [[[1, 2, 'a'], [11, 22, 'b']], [[3, 4, 'c'], [33, 44, 'd']]]\n        In [26]: sum(l, [])\n        Out[26]: [[1, 2, 'a'], [11, 22, 'b'], [3, 4, 'c'], [33, 44, 'd']]\n        "
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores
if __name__ == '__main__':
    dataset = loadDataSet('data/7.RandomForest/sonar-all-data.txt')
    n_folds = 5
    max_depth = 20
    min_size = 1
    sample_size = 1.0
    n_features = 15
    for n_trees in [1, 10, 20]:
        scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        seed(1)
        print('random=', random())
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))