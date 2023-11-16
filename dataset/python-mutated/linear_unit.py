from perceptron import Perceptron
f = lambda x: x

class LinearUnit(Perceptron):
    """
    Desc:
        线性单元类
    Args:
        Perceptron —— 感知器
    Returns:
        None
    """

    def __init__(self, input_num):
        if False:
            for i in range(10):
                print('nop')
        '\n        Desc:\n            初始化线性单元，设置输入参数的个数\n        Args:\n            input_num —— 输入参数的个数\n        Returns:\n            None\n        '
        Perceptron.__init__(self, input_num, f)

def get_training_dataset():
    if False:
        return 10
    '\n    Desc:\n        构建一个简单的训练数据集\n    Args:\n        None\n    Returns:\n        input_vecs —— 训练数据集的特征部分\n        labels —— 训练数据集的数据对应的标签，是一一对应的\n    '
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return (input_vecs, labels)

def train_linear_unit():
    if False:
        while True:
            i = 10
    '\n    Desc:\n        使用训练数据集对我们的线性单元进行训练\n    Args:\n        None\n    Returns:\n        lu —— 返回训练好的线性单元\n    '
    lu = LinearUnit(1)
    (input_vecs, labels) = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    return lu

def plot(linear_unit):
    if False:
        while True:
            i = 10
    '\n    Desc:\n        将我们训练好的线性单元对数据的分类情况作图画出来\n    Args:\n        linear_unit —— 训练好的线性单元\n    Returns:\n        None\n    '
    import matplotlib.pyplot as plt
    (input_vecs, labels) = get_training_dataset()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(list(map(lambda x: x[0], input_vecs)), labels)
    weights = linear_unit.weights
    bias = linear_unit.bias
    y1 = 0 * linear_unit.weights[0] + linear_unit.bias
    y2 = 12 * linear_unit.weights[0] + linear_unit.bias
    plt.plot([0, 12], [y1, y2])
    plt.show()
if __name__ == '__main__':
    '\n    Desc:\n        main 函数，训练我们的线性单元，并进行预测\n    Args:\n        None\n    Returns:\n        None\n    '
    linear_unit = train_linear_unit()
    print(linear_unit)
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
    plot(linear_unit)
from Perceptron import Perceptron
from matplotlib import pyplot as plt
f = lambda x: x

class LinearUnit(Perceptron):

    def __init__(self, input_num):
        if False:
            i = 10
            return i + 15
        '初始化线性单元，设置输入参数的个数'
        Perceptron.__init__(self, input_num, f)

def get_train_dataset():
    if False:
        print('Hello World!')
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return (input_vecs, labels)

def train_linear_unit():
    if False:
        i = 10
        return i + 15
    lu = LinearUnit(1)
    (input_vecs, labels) = get_train_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    return lu
'\n#画图模块\ndef plot(linear_unit):\n    import matplotlib.pyplot as plt\n    input_vecs, labels = get_training_dataset()\n    fig = plt.figure()\n    ax = fig.add_subplot(111)\n    ax.scatter(map(lambda x: x[0], input_vecs), labels)\n    weights = linear_unit.weights\n    bias = linear_unit.bias\n    x = range(0,12,1)\n    y = map(lambda x:weights[0] * x + bias, x)\n    ax.plot(x, y)\n    plt.show()\n'
if __name__ == '__main__':
    linear_unit = train_linear_unit()
    (input_vecs, labels) = get_train_dataset()
    print(linear_unit)
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
    print(linear_unit.weights)
    plt.scatter(input_vecs, labels)
    y1 = 0 * linear_unit.weights[0] + linear_unit.bias
    y2 = 12 * linear_unit.weights[0] + linear_unit.bias
    plt.plot([0, 12], [y1, y2])
    plt.show()