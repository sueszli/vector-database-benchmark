import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def createDataSet():
    if False:
        for i in range(10):
            print('nop')
    ' 数据读入 '
    data = []
    labels = []
    with open('data/3.DecisionTree/data.txt') as ifile:
        for line in ifile:
            tokens = line.strip().split(' ')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    x = np.array(data)
    labels = np.array(labels)
    y = np.zeros(labels.shape)
    ' 标签转换为0/1 '
    y[labels == 'fat'] = 1
    print(data, '-------', x, '-------', labels, '-------', y)
    return (x, y)

def predict_train(x_train, y_train):
    if False:
        i = 10
        return i + 15
    '\n    使用信息熵作为划分标准，对决策树进行训练\n    参考链接:  http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier\n    '
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)
    ' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '
    print('feature_importances_: %s' % clf.feature_importances_)
    '测试结果的打印'
    y_pre = clf.predict(x_train)
    print(y_pre)
    print(y_train)
    print(np.mean(y_pre == y_train))
    return (y_pre, clf)

def show_precision_recall(x, y, clf, y_train, y_pre):
    if False:
        i = 10
        return i + 15
    '\n    准确率与召回率\n    参考链接:  http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve\n    '
    (precision, recall, thresholds) = precision_recall_curve(y_train, y_pre)
    answer = clf.predict_proba(x)[:, 1]
    '\n    展现 准确率与召回率\n        precision 准确率\n        recall 召回率\n        f1-score  准确率和召回率的一个综合得分\n        support 参与比较的数量\n    参考链接: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report\n    '
    target_names = ['thin', 'fat']
    print(classification_report(y, answer, target_names=target_names))
    print(answer)
    print(y)

def show_pdf(clf):
    if False:
        while True:
            i = 10
    "\n    可视化输出\n    把决策树结构写入文件: http://sklearn.lzjqsdd.com/modules/tree.html\n\n    Mac报错: pydotplus.graphviz.InvocationException: GraphViz's executables not found\n    解决方案: sudo brew install graphviz\n    参考写入:  http://www.jianshu.com/p/59b510bafb4d\n    "
    import pydotplus
    from sklearn.externals.six import StringIO
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('../../../output/3.DecisionTree/tree.pdf')
if __name__ == '__main__':
    (x, y) = createDataSet()
    ' 拆分训练数据与测试数据， 80%做训练 20%做测试 '
    (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2)
    print('拆分数据: ', x_train, x_test, y_train, y_test)
    (y_pre, clf) = predict_train(x_train, y_train)
    show_precision_recall(x, y, clf, y_train, y_pre)
    show_pdf(clf)