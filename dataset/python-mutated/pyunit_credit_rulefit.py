import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.rulefit import H2ORuleFitEstimator

def credit():
    if False:
        for i in range(10):
            print('nop')
    df = h2o.import_file(pyunit_utils.locate('smalldata/kaggle/CreditCard/creditcard_train_cat.csv'))
    df['DEFAULT_PAYMENT_NEXT_MONTH'] = df['DEFAULT_PAYMENT_NEXT_MONTH'].asfactor()
    x = df.columns
    y = 'DEFAULT_PAYMENT_NEXT_MONTH'
    x.remove(y)
    (train, test) = df.split_frame(ratios=[0.8], seed=1234)
    rfit = H2ORuleFitEstimator(min_rule_length=1, max_rule_length=3, max_num_rules=10, seed=1234, model_type='rules_and_linear')
    rfit.train(training_frame=train, x=x, y=y, validation_frame=test)
    python_lists = [[0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]]
    h2oframe = h2o.H2OFrame(python_obj=python_lists, column_names=df.names, column_types=df.types, na_strings=['NA'])
    df = df.concat(h2oframe, 0)
    (train, test) = df.split_frame(ratios=[0.8], seed=1234)
    rfit_multi = H2ORuleFitEstimator(min_rule_length=1, max_rule_length=3, max_num_rules=10, seed=1234, model_type='rules_and_linear')
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
if __name__ == '__main__':
    pyunit_utils.standalone_test(credit)
else:
    credit()