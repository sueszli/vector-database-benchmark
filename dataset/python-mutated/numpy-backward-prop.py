"""
@author: HuRuiFeng
@file: 7.9-backward-prop.py
@time: 2020/2/24 17:32
@desc: 7.9 反向传播算法实战的代码
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False

def load_dataset():
    if False:
        while True:
            i = 10
    N_SAMPLES = 2000
    TEST_SIZE = 0.3
    (X, y) = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    return (X, y, X_train, X_test, y_train, y_test)

def make_plot(X, y, plot_name, XX=None, YY=None, preds=None, dark=False):
    if False:
        while True:
            i = 10
    if dark:
        plt.style.use('dark_background')
    else:
        sns.set_style('whitegrid')
    plt.figure(figsize=(16, 12))
    axes = plt.gca()
    axes.set(xlabel='$x_1$', ylabel='$x_2$')
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(right=0.8)
    if XX is not None and YY is not None and (preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=1, cmap=plt.cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[0.5], cmap='Greys', vmin=0, vmax=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='none')
    plt.savefig('数据集分布.svg')
    plt.close()

class Layer:

    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        if False:
            i = 10
            return i + 15
        '\n        :param int n_input: 输入节点数\n        :param int n_neurons: 输出节点数\n        :param str activation: 激活函数类型\n        :param weights: 权值张量，默认类内部生成\n        :param bias: 偏置，默认类内部生成\n        '
        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.1
        self.activation = activation
        self.last_activation = None
        self.error = None
        self.delta = None

    def activate(self, x):
        if False:
            i = 10
            return i + 15
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        if False:
            print('Hello World!')
        if self.activation is None:
            return r
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

    def apply_activation_derivative(self, r):
        if False:
            print('Hello World!')
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.0
            grad[r <= 0] = 0.0
            return grad
        elif self.activation == 'tanh':
            return 1 - r ** 2
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r

class NeuralNetwork:

    def __init__(self):
        if False:
            print('Hello World!')
        self._layers = []

    def add_layer(self, layer):
        if False:
            for i in range(10):
                print('nop')
        self._layers.append(layer)

    def feed_forward(self, X):
        if False:
            return 10
        for layer in self._layers:
            X = layer.activate(X)
        return X

    def backpropagation(self, X, y, learning_rate):
        if False:
            i = 10
            return i + 15
        output = self.feed_forward(X)
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer == self._layers[-1]:
                layer.error = y - output
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
        for i in range(len(self._layers)):
            layer = self._layers[i]
            o_i = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * o_i.T * learning_rate

    def train(self, X_train, X_test, y_train, y_test, learning_rate, max_epochs):
        if False:
            return 10
        y_onehot = np.zeros((y_train.shape[0], 2))
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        mses = []
        accuracys = []
        for i in range(max_epochs + 1):
            for j in range(len(X_train)):
                self.backpropagation(X_train[j], y_onehot[j], learning_rate)
            if i % 10 == 0:
                mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
                mses.append(mse)
                accuracy = self.accuracy(self.predict(X_test), y_test.flatten())
                accuracys.append(accuracy)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))
                print('Accuracy: %.2f%%' % (accuracy * 100))
        return (mses, accuracys)

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        return self.feed_forward(X)

    def accuracy(self, X, y):
        if False:
            while True:
                i = 10
        return np.sum(np.equal(np.argmax(X, axis=1), y)) / y.shape[0]

def main():
    if False:
        return 10
    (X, y, X_train, X_test, y_train, y_test) = load_dataset()
    make_plot(X, y, 'Classification Dataset Visualization ')
    plt.show()
    nn = NeuralNetwork()
    nn.add_layer(Layer(2, 25, 'sigmoid'))
    nn.add_layer(Layer(25, 50, 'sigmoid'))
    nn.add_layer(Layer(50, 25, 'sigmoid'))
    nn.add_layer(Layer(25, 2, 'sigmoid'))
    (mses, accuracys) = nn.train(X_train, X_test, y_train, y_test, 0.01, 1000)
    x = [i for i in range(0, 101, 10)]
    plt.title('MES Loss')
    plt.plot(x, mses[:11], color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig('训练误差曲线.svg')
    plt.close()
    plt.title('Accuracy')
    plt.plot(x, accuracys[:11], color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('网络测试准确率.svg')
    plt.close()
if __name__ == '__main__':
    main()