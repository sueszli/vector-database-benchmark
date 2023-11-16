import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator as H2OPCA

def glrm_iris():
    if False:
        for i in range(10):
            print('nop')
    print('Importing iris.csv data...')
    irisH2O = h2o.upload_file(pyunit_utils.locate('smalldata/iris/iris.csv'))
    irisH2O.describe()
    print('@@@@@@  Building PCA with GramSVD...\n')
    glrmPCA = H2OPCA(k=5, transform='STANDARDIZE', pca_method='GLRM', use_all_factor_levels=True, seed=21)
    glrmPCA.train(x=irisH2O.names, training_frame=irisH2O)
    glrm_h2o = H2OGeneralizedLowRankEstimator(k=5, loss='Quadratic', transform='STANDARDIZE', recover_svd=True, seed=21)
    glrm_h2o.train(x=irisH2O.names, training_frame=irisH2O)
    print('@@@@@@  Comparing eigenvalues between GramSVD and GLRM...\n')
    pyunit_utils.assert_H2OTwoDimTable_equal(glrmPCA._model_json['output']['importance'], glrm_h2o._model_json['output']['importance'], ['Standard deviation', 'Cumulative Proportion', 'Cumulative Proportion'], tolerance=1e-06)
    print('@@@@@@  Comparing eigenvectors between GramSVD and GLRM...\n')
    pyunit_utils.assert_H2OTwoDimTable_equal(glrmPCA._model_json['output']['eigenvectors'], glrm_h2o._model_json['output']['eigenvectors'], glrm_h2o._model_json['output']['names'], tolerance=1e-06, check_sign=True)
    assert glrmPCA._model_json['output']['importance'].cell_values[1][1] <= 1, 'Expected value <= 1.0 but received {0}'.format(glrmPCA._model_json['output']['importance'].cell_values[1][1])
if __name__ == '__main__':
    pyunit_utils.standalone_test(glrm_iris)
else:
    glrm_iris()