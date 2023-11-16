from __future__ import print_function, division
from builtins import range
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from theano.tensor.shared_randomstreams import RandomStreams
from util import relu, error_rate, getKaggleMNIST, init_weights
from autoencoder import AutoEncoder, momentum_updates
from rbm import RBM

class DBN(object):

    def __init__(self, hidden_layer_sizes, UnsupervisedModel=AutoEncoder):
        if False:
            print('Hello World!')
        self.hidden_layers = []
        count = 0
        for M in hidden_layer_sizes:
            ae = UnsupervisedModel(M, count)
            self.hidden_layers.append(ae)
            count += 1

    def fit(self, X, pretrain_epochs=1):
        if False:
            i = 10
            return i + 15
        self.D = X.shape[1]
        current_input = X
        for ae in self.hidden_layers:
            ae.fit(current_input, epochs=pretrain_epochs)
            current_input = ae.hidden_op(current_input)
        return current_input

    def forward(self, X):
        if False:
            i = 10
            return i + 15
        Z = X
        for ae in self.hidden_layers:
            Z = ae.forward_hidden(Z)
        return Z

    def fit_to_input(self, k, learning_rate=1.0, mu=0.99, epochs=100000):
        if False:
            while True:
                i = 10
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)
        X0 = init_weights((1, self.D))
        X = theano.shared(X0, 'X_shared')
        Y = self.forward(X)
        cost = -T.log(Y[0, k])
        updates = momentum_updates(cost, [X], mu, learning_rate)
        train = theano.function(inputs=[], outputs=[cost, Y], updates=updates)
        costs = []
        for i in range(epochs):
            if i % 10000 == 0:
                print('epoch:', i)
            (the_cost, out) = train()
            if i == 0:
                print('out.shape:', out.shape)
            costs.append(the_cost)
        plt.plot(costs)
        plt.show()
        return X.get_value()

    def save(self, filename):
        if False:
            i = 10
            return i + 15
        arrays = [p.get_value() for layer in self.hidden_layers for p in layer.params]
        np.savez(filename, *arrays)

    @staticmethod
    def load(filename, UnsupervisedModel=AutoEncoder):
        if False:
            while True:
                i = 10
        dbn = DBN([], UnsupervisedModel)
        npz = np.load(filename)
        dbn.hidden_layers = []
        count = 0
        for i in range(0, len(npz.files), 3):
            W = npz['arr_%s' % i]
            bh = npz['arr_%s' % (i + 1)]
            bo = npz['arr_%s' % (i + 2)]
            if i == 0:
                dbn.D = W.shape[0]
            ae = UnsupervisedModel.createFromArrays(W, bh, bo, count)
            dbn.hidden_layers.append(ae)
            count += 1
        return dbn

def main():
    if False:
        return 10
    (Xtrain, Ytrain, Xtest, Ytest) = getKaggleMNIST()
    dbn = DBN([1000, 750, 500], UnsupervisedModel=AutoEncoder)
    output = dbn.fit(Xtrain, pretrain_epochs=2)
    print('output.shape', output.shape)
    sample_size = 600
    tsne = TSNE()
    reduced = tsne.fit_transform(output[:sample_size])
    plt.scatter(reduced[:, 0], reduced[:, 1], s=100, c=Ytrain[:sample_size], alpha=0.5)
    plt.title('t-SNE visualization on data transformed by DBN')
    plt.show()
    reduced = tsne.fit_transform(Xtrain[:sample_size])
    plt.scatter(reduced[:, 0], reduced[:, 1], s=100, c=Ytrain[:sample_size], alpha=0.5)
    plt.title('t-SNE visualization on raw data')
    plt.show()
    pca = PCA()
    reduced = pca.fit_transform(output)
    plt.scatter(reduced[:, 0], reduced[:, 1], s=100, c=Ytrain, alpha=0.5)
    plt.title('PCA visualization on data transformed by DBN')
    plt.show()
if __name__ == '__main__':
    main()