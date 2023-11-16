import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.rulefit import H2ORuleFitEstimator

def cancer():
    if False:
        print('Hello World!')
    df = h2o.import_file('https://h2o-public-test-data.s3.amazonaws.com/mli-testing/manual-test/small-dataset/binomial/risk_factors_cervical_cancer.csv')
    df['Biopsy'] = df['Biopsy'].asfactor()
    x = df.columns
    y = 'Biopsy'
    x.remove(y)
    (train, test) = df.split_frame(ratios=[0.8], seed=1234)
    rfit = H2ORuleFitEstimator(min_rule_length=1, max_rule_length=10, max_num_rules=100, seed=1234, model_type='rules')
    rfit.train(training_frame=train, x=x, y=y, validation_frame=test)
    python_lists = [[0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]]
    h2oframe = h2o.H2OFrame(python_obj=python_lists, column_names=df.names, column_types=df.types, na_strings=['NA'])
    df = df.concat(h2oframe, 0)
    (train, test) = df.split_frame(ratios=[0.8], seed=1234)
    rfit_multi = H2ORuleFitEstimator(min_rule_length=1, max_rule_length=10, max_num_rules=100, seed=1234, model_type='rules')
    rfit_multi.train(training_frame=train, x=x, y=y, validation_frame=test)
    print('Binomial model rules:')
    print(rfit.rule_importance())
    print('Multinomial model rules:')
    print(rfit_multi.rule_importance())
    print('Binomial train RMSE vs. multinomial train RMSE:')
    print(str(rfit.rmse()) + ' vs. ' + str(rfit_multi.rmse()))
    print('Binomial train MSE vs. multinomial train MSE: ')
    print(str(rfit.mse()) + ' vs. ' + str(rfit_multi.mse()))
    print('Binomial valid RMSE vs. multinomial valid RMSE: ')
    print(str(rfit.rmse(valid=True)) + ' vs. ' + str(rfit_multi.rmse(valid=True)))
    print('Binomial valid MSE vs. multinomial valid MSE: ')
    print(str(rfit.mse(valid=True)) + ' vs. ' + str(rfit_multi.mse(valid=True)))
    rfit_multi = H2ORuleFitEstimator(min_rule_length=1, max_rule_length=10, max_num_rules=100, seed=1234, model_type='rules', max_categorical_levels=3)
    rfit_multi.train(training_frame=train, x=x, y=y, validation_frame=test)
    print(rfit_multi.rule_importance())
if __name__ == '__main__':
    pyunit_utils.standalone_test(cancer)
else:
    cancer()