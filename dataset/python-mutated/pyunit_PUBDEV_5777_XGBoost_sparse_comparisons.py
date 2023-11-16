import sys
sys.path.insert(1, '../../../')
from h2o.estimators.xgboost import *
from tests import pyunit_utils
'\nPUBDEV-5777: enable H2OXGBoost and native XGBoost comparison.\n\nTo ensure that H2OXGBoost and native XGBoost performance provide the same result, we propose to do the following:\n1. run H2OXGBoost with H2OFrame and parameter setting, save model and result\n2. Call Python API to convert H2OFrame to XGBoost frame and H2OXGBoost parameter to XGBoost parameters.\n3. Run native XGBoost with data frame and parameters from 2.  Should get the same result as in 1.\n\nParameters in native XGBoost:\nbooster default to gbtree\nsilent default to 0\nnthread default to maximum number of threads available if not specified\ndisable_default_eval_metric default to 0\nnum_pbuffer automatically set\nnum_feature automatically set\neta/learning_rate default to 0.3\nmax_depth default to 6\nmin_child_weight default to 1\nmax_delta_step default to 0\nsubsample default to 1\ncolsample_bytree default to 1\ncolsample_by_level default to 1\nlambda/reg_lambda default to 1\nalpha/reg_alpha default to 0\ntree_method default to auto\nsketch_eps default to 0.03\nscale_pos_weight default to 1\nupdater default to grow_colmaker, prune\nrefresh_leaf default to 1\nprocess_type default to default\ngrow_policy default depthwise\nmax_leaves default to 0\nmax_bin default to 256\npredictor default to cpu_predictor\n\nAddition ones for DART booster\nsmaple_type default to uniform\nnormalize_type default to tree\nrate_drop default to 0.0\none_drop default to 0.0\nskip_drop default to 0.0\n\nFor Linear Booster\nlambda/reg_lambda default to 0\nalpha/reg_alpha default to 0\nupdater default to shotgun\nfeature_selector default to cyclic\ntop_k default to 0\n\nParameters for Tweedie Regression objective=reg:tweedie\ntweedie_variance_power default to 1.5\n\nlearning Task parameters:\nobjective default to reg:linear\nbase_score default to 0.5\neval_metric default according to objective\nseed default to 0\n'

def comparison_test():
    if False:
        while True:
            i = 10
    if sys.version.startswith('2'):
        print('native XGBoost tests only supported on python3')
        return
    import xgboost as xgb
    assert H2OXGBoostEstimator.available() is True
    ret = h2o.cluster()
    if len(ret.nodes) == 1:
        runSeed = 1
        dataSeed = 17
        ntrees = 17
        maxdepth = 5
        nrows = 10000
        ncols = 12
        factorL = 20
        numCols = 1
        enumCols = ncols - numCols
        responseL = 4
        h2oParamsD = {'ntrees': ntrees, 'max_depth': maxdepth, 'seed': runSeed, 'learn_rate': 0.7, 'col_sample_rate_per_tree': 0.9, 'min_rows': 5, 'score_tree_interval': ntrees + 1, 'tree_method': 'exact', 'backend': 'cpu'}
        trainFile = pyunit_utils.genTrainFrame(nrows, numCols, enumCols=enumCols, enumFactors=factorL, responseLevel=responseL, miscfrac=0.01, randseed=dataSeed)
        myX = trainFile.names
        y = 'response'
        myX.remove(y)
        newNames = []
        for ind in range(0, len(myX)):
            myX[ind] = myX[ind] + '_' + str(ind)
            newNames.append(myX[ind])
        newNames.append(y)
        trainFile.set_names(newNames)
        h2oModelD = H2OXGBoostEstimator(**h2oParamsD)
        h2oModelD.train(x=myX, y=y, training_frame=trainFile)
        h2oPredictD = h2oModelD.predict(trainFile)
        nativeXGBoostParam = h2oModelD.convert_H2OXGBoostParams_2_XGBoostParams()
        nativeXGBoostInput = trainFile.convert_H2OFrame_2_DMatrix(myX, y, h2oModelD)
        nativeModel = xgb.train(params=nativeXGBoostParam[0], dtrain=nativeXGBoostInput, num_boost_round=nativeXGBoostParam[1])
        nativePred = nativeModel.predict(data=nativeXGBoostInput, ntree_limit=nativeXGBoostParam[1])
        pyunit_utils.summarizeResult_multinomial(h2oPredictD, nativePred, -1, -1, -1, -1, tolerance=1e-06)
    else:
        print('********  Test skipped.  This test cannot be performed in multinode environment.')
if __name__ == '__main__':
    pyunit_utils.standalone_test(comparison_test)
else:
    comparison_test()