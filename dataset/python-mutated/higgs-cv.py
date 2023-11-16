import numpy as np
import xgboost as xgb
train = np.loadtxt('./data/training.csv', delimiter=',', skiprows=1, converters={32: lambda x: int(x == 's'.encode('utf-8'))})
label = train[:, 32]
data = train[:, 1:31]
weight = train[:, 31]
dtrain = xgb.DMatrix(data, label=label, missing=-999.0, weight=weight)
param = {'max_depth': 6, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logitraw', 'nthread': 4}
num_round = 120
print('running cross validation, with preprocessing function')

def fpreproc(dtrain, dtest, param):
    if False:
        for i in range(10):
            print('nop')
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    wtrain = dtrain.get_weight()
    wtest = dtest.get_weight()
    sum_weight = sum(wtrain) + sum(wtest)
    wtrain *= sum_weight / sum(wtrain)
    wtest *= sum_weight / sum(wtest)
    dtrain.set_weight(wtrain)
    dtest.set_weight(wtest)
    return (dtrain, dtest, param)
xgb.cv(param, dtrain, num_round, nfold=5, metrics={'ams@0.15', 'auc'}, seed=0, fpreproc=fpreproc)