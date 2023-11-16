import sys
sys.path.insert(1, '../../../')
import h2o
from random import randint
from tests import pyunit_utils
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator as H2OPCA
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator

def pca_wideDataset_rotterdam_pcapower():
    if False:
        for i in range(10):
            print('nop')
    tol = 2e-05
    h2o.remove_all()
    print('Importing Rotterdam.csv data...')
    rotterdamH2O = h2o.upload_file(pyunit_utils.locate('bigdata/laptop/jira/rotterdam.csv.zip'))
    y = set(['relapse'])
    x = list(set(rotterdamH2O.names) - y)
    print('------  Testing Power PCA --------')
    gramSVD = H2OPCA(k=8, impute_missing=True, transform='STANDARDIZE', seed=12345)
    gramSVD.train(x=x, training_frame=rotterdamH2O)
    powerPCA = H2OPCA(k=8, impute_missing=True, transform='STANDARDIZE', pca_method='Power', seed=12345)
    powerPCA.train(x=x, training_frame=rotterdamH2O)
    print('@@@@@@  Comparing eigenvalues between GramSVD and Power...\n')
    pyunit_utils.assert_H2OTwoDimTable_equal(gramSVD._model_json['output']['importance'], powerPCA._model_json['output']['importance'], ['Standard deviation', 'Cumulative Proportion', 'Cumulative Proportion'], tolerance=1e-06, check_all=False)
    print('@@@@@@  Comparing eigenvectors between GramSVD and Power...\n')
    pyunit_utils.assert_H2OTwoDimTable_equal(gramSVD._model_json['output']['eigenvectors'], powerPCA._model_json['output']['eigenvectors'], powerPCA._model_json['output']['names'], tolerance=tol, check_sign=True, check_all=False)
if __name__ == '__main__':
    pyunit_utils.standalone_test(pca_wideDataset_rotterdam_pcapower)
else:
    pca_wideDataset_rotterdam_pcapower()