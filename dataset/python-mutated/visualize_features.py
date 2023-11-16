from __future__ import print_function, division
from builtins import range, input
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from theano.tensor.shared_randomstreams import RandomStreams
from util import relu, error_rate, getKaggleMNIST, init_weights
from unsupervised import DBN
from rbm import RBM

def main(loadfile=None, savefile=None):
    if False:
        i = 10
        return i + 15
    (Xtrain, Ytrain, Xtest, Ytest) = getKaggleMNIST()
    if loadfile:
        dbn = DBN.load(loadfile)
    else:
        dbn = DBN([1000, 750, 500, 10])
        dbn.fit(Xtrain, pretrain_epochs=2)
    if savefile:
        dbn.save(savefile)
    W = dbn.hidden_layers[0].W.eval()
    for i in range(dbn.hidden_layers[0].M):
        imgplot = plt.imshow(W[:, i].reshape(28, 28), cmap='gray')
        plt.show()
        should_quit = input("Show more? Enter 'n' to quit\n")
        if should_quit == 'n':
            break
    for k in range(dbn.hidden_layers[-1].M):
        X = dbn.fit_to_input(k)
        imgplot = plt.imshow(X.reshape(28, 28), cmap='gray')
        plt.show()
        if k < dbn.hidden_layers[-1].M - 1:
            should_quit = input("Show more? Enter 'n' to quit\n")
            if should_quit == 'n':
                break
if __name__ == '__main__':
    main()