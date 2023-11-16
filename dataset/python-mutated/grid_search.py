from __future__ import print_function, division
from builtins import range
import theano.tensor as T
from theano_ann import ANN
from util import get_spiral, get_clouds
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

def grid_search():
    if False:
        i = 10
        return i + 15
    (X, Y) = get_spiral()
    (X, Y) = shuffle(X, Y)
    Ntrain = int(0.7 * len(X))
    (Xtrain, Ytrain) = (X[:Ntrain], Y[:Ntrain])
    (Xtest, Ytest) = (X[Ntrain:], Y[Ntrain:])
    hidden_layer_sizes = [[300], [100, 100], [50, 50, 50]]
    learning_rates = [0.0001, 0.001, 0.01]
    l2_penalties = [0.0, 0.1, 1.0]
    best_validation_rate = 0
    best_hls = None
    best_lr = None
    best_l2 = None
    for hls in hidden_layer_sizes:
        for lr in learning_rates:
            for l2 in l2_penalties:
                model = ANN(hls)
                model.fit(Xtrain, Ytrain, learning_rate=lr, reg=l2, mu=0.99, epochs=3000, show_fig=False)
                validation_accuracy = model.score(Xtest, Ytest)
                train_accuracy = model.score(Xtrain, Ytrain)
                print('validation_accuracy: %.3f, train_accuracy: %.3f, settings: %s, %s, %s' % (validation_accuracy, train_accuracy, hls, lr, l2))
                if validation_accuracy > best_validation_rate:
                    best_validation_rate = validation_accuracy
                    best_hls = hls
                    best_lr = lr
                    best_l2 = l2
    print('Best validation_accuracy:', best_validation_rate)
    print('Best settings:')
    print('hidden_layer_sizes:', best_hls)
    print('learning_rate:', best_lr)
    print('l2:', best_l2)
if __name__ == '__main__':
    grid_search()