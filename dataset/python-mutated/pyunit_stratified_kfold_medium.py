from builtins import range
from past.utils import old_div
import sys
from functools import reduce
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def stratified_kfold():
    if False:
        i = 10
        return i + 15
    NFOLDS = 5
    fr = h2o.import_file(pyunit_utils.locate('bigdata/laptop/covtype/covtype.data'))
    stratified = fr[54].stratified_kfold_column(n_folds=NFOLDS)
    stratified.show()
    dist = old_div(fr[54].table()['Count'], fr[54].table()['Count'].sum()).as_data_frame(True).to_dict('list')['Count']
    overall_result = reduce(lambda x, y: x.cbind(y), [old_div(fr[stratified == i, 54].table()['Count'], fr[stratified == i, 54].table()['Count'].sum()) for i in range(NFOLDS)])
    overall_result.show()
    df = overall_result.as_data_frame(True)
    print()
    print('Show that all folds are consistent with one another: ')
    print(df.mean(axis=1))
    print(df.var(axis=1))
    print()
    for i in range(len(dist)):
        print('Stratification variance for class #%s: %s' % (i, old_div(df.loc[i].sub(dist[i]).pow(2).sum(), df.shape[0] - 1)))
if __name__ == '__main__':
    pyunit_utils.standalone_test(stratified_kfold)
else:
    stratified_kfold()