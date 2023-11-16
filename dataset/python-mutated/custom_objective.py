import numpy as np
import xgboost as xgb
print('start running example to used cutomized objective function')
dtrain = xgb.DMatrix('../data/agaricus.txt.train')
dtest = xgb.DMatrix('../data/agaricus.txt.test')
param = {'max_depth': 2, 'eta': 1, 'silent': 1}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 2

def logregobj(preds, dtrain):
    if False:
        while True:
            i = 10
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return (grad, hess)

def evalerror(preds, dtrain):
    if False:
        return 10
    labels = dtrain.get_label()
    return ('error', float(sum(labels != (preds > 0.0))) / len(labels))
bst = xgb.train(param, dtrain, num_round, watchlist, logregobj, evalerror)