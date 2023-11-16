from builtins import range
import sys, os
sys.path.insert(1, os.path.join('..', '..'))
import h2o
from tests import pyunit_utils, assert_equals
from h2o.estimators import H2OGradientBoostingEstimator, H2OXGBoostEstimator
import numpy as np
import pandas as pd
import imp

def prepare_data():
    if False:
        i = 10
        return i + 15
    '\n    Generate data with target variable: \n\n    p(Y) = 1/ (1 + exp(-(-3 + 0.5X1 + 0.5X2 - 0.5X3 + 2X2X3)))\n\n    :return: Dataframe, X, and y\n    '
    n = 1000
    np.random.seed(0)
    X1 = np.random.normal(0, 1, size=n)
    np.random.seed(1)
    X2 = np.random.normal(0, 1, size=n)
    np.random.seed(2)
    X3 = np.random.normal(0, 1, size=n)
    np.random.seed(3)
    R = np.random.uniform(0, 1, size=n)
    df = pd.DataFrame({'id': range(1, n + 1), 'X1': X1, 'X2': X2, 'X3': X3, 'R': R})
    B0 = -3
    B1 = 0.5
    B2 = 0.5
    B3 = -0.5
    B12 = 0
    B13 = 0
    B23 = 2
    df['LP'] = B0 + B1 * df['X1'] + B2 * df['X2'] + B3 * df['X3'] + B12 * df['X1'] * df['X2'] + B13 * df['X1'] * df['X3'] + B23 * df['X2'] * df['X3']
    df['P'] = 1 / (1 + np.exp(-df['LP']))
    df['Y'] = (df['R'] < df['P']).astype(int)
    X = ['X1', 'X2', 'X3']
    y = 'Y'
    return (df, X, y)

def provide_sklearn_output_if_possible(df, x, target):
    if False:
        print('Hello World!')
    try:
        imp.find_module('sklearn_gbmi')
        sklearn_gbmi_found = True
    except ImportError:
        sklearn_gbmi_found = False
    x1x2_sklearn = 0.08209119711536361
    x1x3_sklearn = 0.14613801828239015
    x2x3_sklearn = 0.6938075033110082
    if sklearn_gbmi_found:
        print('Obtaining actual sklearn output')
        from sklearn_gbmi import h_all_pairs
        from sklearn.ensemble import GradientBoostingClassifier
        X = df[x]
        y = df[[target]]
        gbm_sklearn = GradientBoostingClassifier(n_estimators=100, random_state=1234, max_depth=2, learning_rate=0.1, min_samples_leaf=1)
        gbm_sklearn.fit(X, np.ravel(y))
        sklearn_h = h_all_pairs(gbm_sklearn, X)
        x1x2_sklearn = sklearn_h.get(('X1', 'X2'))
        x1x3_sklearn = sklearn_h.get(('X1', 'X3'))
        x2x3_sklearn = sklearn_h.get(('X2', 'X3'))
    return (x1x2_sklearn, x1x3_sklearn, x2x3_sklearn)

def h_stats_on_synthetic_data():
    if False:
        return 10
    (df, x, target) = prepare_data()
    train_frame = h2o.H2OFrame(df[x + [target]])
    train_frame[target] = train_frame[target].asfactor()
    gbm_h2o = H2OGradientBoostingEstimator(ntrees=100, learn_rate=0.1, max_depth=2, min_rows=1, seed=1234)
    gbm_h2o.train(x=x, y=target, training_frame=train_frame)
    xgb_h2o = H2OXGBoostEstimator(ntrees=100, learn_rate=0.1, max_depth=2, min_rows=1, seed=1234)
    xgb_h2o.train(x=x, y=target, training_frame=train_frame)
    (x1x2_sklearn, x1x3_sklearn, x2x3_sklearn) = provide_sklearn_output_if_possible(df, x, target)
    print("Sci-GBM H: ('X1', 'X2')", x1x2_sklearn, "('X1', 'X3')", x1x3_sklearn, "('X2', 'X3')", x2x3_sklearn)
    (x1x2_gbm, x1x3_gbm, x2x3_gbm) = (gbm_h2o.h(train_frame, ['X1', 'X2']), gbm_h2o.h(train_frame, ['X1', 'X3']), gbm_h2o.h(train_frame, ['X2', 'X3']))
    print("H2O-GBM H: ('X1', 'X2')", x1x2_gbm, "('X1', 'X3')", x1x3_gbm, "('X2', 'X3')", x2x3_gbm)
    (x1x2_xgb, x1x3_xgb, x2x3_xgb) = (xgb_h2o.h(train_frame, ['X1', 'X2']), xgb_h2o.h(train_frame, ['X1', 'X3']), xgb_h2o.h(train_frame, ['X2', 'X3']))
    print("H2O-XGB H: ('X1', 'X2')", x1x2_xgb, "('X1', 'X3')", x1x3_xgb, "('X2', 'X3')", x2x3_xgb)
    assert_equals(x1x2_sklearn, x1x2_gbm, 'Not expected output: H stats should be around sklearn reference', 0.1)
    assert_equals(x1x3_sklearn, x1x3_gbm, 'Not expected output: H stats should be around sklearn reference', 0.1)
    assert_equals(x2x3_sklearn, x2x3_gbm, 'Not expected output: H stats should be around sklearn reference', 0.1)
    assert_equals(x1x2_sklearn, x1x2_xgb, 'Not expected output: H stats should be around sklearn reference', 0.1)
    assert_equals(x1x3_sklearn, x1x3_xgb, 'Not expected output: H stats should be around sklearn reference', 0.1)
    assert_equals(x2x3_sklearn, x2x3_xgb, 'Not expected output: H stats should be around sklearn reference', 0.1)
if __name__ == '__main__':
    pyunit_utils.standalone_test(h_stats_on_synthetic_data)
else:
    h_stats_on_synthetic_data()