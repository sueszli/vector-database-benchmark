import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
from h2o.assembly import *
from h2o.transforms.preprocessing import *

def lending_club_munging_assembly():
    if False:
        return 10
    small_test = pyunit_utils.locate('bigdata/laptop/lending-club/LoanStats3a.csv')
    print('Import and Parse data')
    types = {'int_rate': 'string', 'revol_util': 'string', 'emp_length': 'string', 'earliest_cr_line': 'string', 'issue_d': 'string', 'last_credit_pull_d': 'factor'}
    data = h2o.import_file(path=small_test, col_types=types)
    data[['int_rate', 'revol_util', 'emp_length']].show()
    assembly = H2OAssembly(steps=[('intrate_rm_junk_char', H2OColOp(op=H2OFrame.gsub, col='int_rate', inplace=True, pattern='%', replacement='')), ('intrate_trim_ws', H2OColOp(op=H2OFrame.trim, col='int_rate', inplace=True)), ('intrate_as_numeric', H2OColOp(op=H2OFrame.asnumeric, col='int_rate', inplace=True)), ('revol_rm_junk_char', H2OColOp(op=H2OFrame.gsub, col='revol_util', inplace=True, pattern='%', replacement='')), ('revol_trim_ws', H2OColOp(op=H2OFrame.trim, col='revol_util', inplace=True)), ('revol_as_numeric', H2OColOp(op=H2OFrame.asnumeric, col='revol_util', inplace=True)), ('earliest_cr_line_split', H2OColOp(H2OFrame.strsplit, col='earliest_cr_line', inplace=False, new_col_name=['earliest_cr_line_Month', 'earliest_cr_line_Year'], pattern='-')), ('earliest_cr_line_Year_as_numeric', H2OColOp(op=H2OFrame.asnumeric, col='earliest_cr_line_Year', inplace=True)), ('issue_date_split', H2OColOp(op=H2OFrame.strsplit, col='issue_d', inplace=False, new_col_name=['issue_d_Month', 'issue_d_Year'], pattern='-')), ('issue_d_Year_as_numeric', H2OColOp(op=H2OFrame.asnumeric, col='issue_d_Year', inplace=True)), ('emp_length_rm_years', H2OColOp(op=H2OFrame.gsub, col='emp_length', inplace=True, pattern='([ ]*+[a-zA-Z].*)|(n/a)', replacement='')), ('emp_length_trim', H2OColOp(op=H2OFrame.trim, col='emp_length', inplace=True)), ('emp_length_lt1_point5', H2OColOp(op=H2OFrame.gsub, col='emp_length', inplace=True, pattern='< 1', replacement='0.5')), ('emp_length_10plus', H2OColOp(op=H2OFrame.gsub, col='emp_length', inplace=True, pattern='10\\+', replacement='10')), ('emp_length_as_numeric', H2OColOp(op=H2OFrame.asnumeric, col='emp_length', inplace=True)), ('credit_length', H2OBinaryOp(op=H2OAssembly.minus, col='issue_d_Year', inplace=False, new_col_name='longest_credit_length', right=H2OCol('earliest_cr_line_Year')))])
    res = assembly.fit(data)
    res.show()
    assembly.to_pojo('LendingClubMungingDemo')
    y = 'int_rate'
    x = ['loan_amnt', 'earliest_cr_line', 'revol_util', 'emp_length', 'home_ownership', 'annual_inc', 'purpose', 'addr_state', 'dti', 'delinq_2yrs', 'total_acc', 'verification_status', 'term']
    from h2o.estimators.gbm import H2OGradientBoostingEstimator
    model = H2OGradientBoostingEstimator(model_id='InterestRateModel', score_each_iteration=False, ntrees=100, max_depth=5, learn_rate=0.05)
    model.train(x=x, y=y, training_frame=data)
    model.download_pojo()
if __name__ == '__main__':
    pyunit_utils.standalone_test(lending_club_munging_assembly)
else:
    lending_club_munging_assembly()