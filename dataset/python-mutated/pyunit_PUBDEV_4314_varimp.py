import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator as H2OPCA
from h2o.utils.typechecks import assert_is_type
from pandas import DataFrame

def pca_pubdev_4314():
    if False:
        print('Hello World!')
    print('Importing prostate_cat.csv data...\n')
    prostate = h2o.upload_file(pyunit_utils.locate('smalldata/prostate/prostate_cat.csv'))
    prostate.describe()
    print("PCA with k = 3, retx = FALSE, transform = 'STANDARDIZE'")
    fitPCA = H2OPCA(k=3, transform='StANDARDIZE', pca_method='GramSVD')
    fitPCA.train(x=list(range(0, 8)), training_frame=prostate)
    print(fitPCA.summary())
    varimpPandas = fitPCA.varimp(use_pandas=True)
    assert_is_type(varimpPandas, DataFrame)
    varimpList = fitPCA.varimp()
    print(varimpList)
    assert_is_type(varimpList, list)
    sys.stdout.flush()
pyunit_utils.standalone_test(pca_pubdev_4314)