import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
from h2o.transforms.preprocessing import H2OScaler
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.pipeline import Pipeline

def scale_svd_rf_pipe():
    if False:
        for i in range(10):
            print('nop')
    from h2o.transforms.decomposition import H2OSVD
    print('Importing USArrests.csv data...')
    arrests = h2o.upload_file(pyunit_utils.locate('smalldata/pca_test/USArrests.csv'))
    pipe = Pipeline([('standardize', H2OScaler()), ('svd', H2OSVD(nv=3)), ('rf', H2ORandomForestEstimator(seed=42, ntrees=50))])
    pipe.fit(arrests[1:], arrests[0])
    print(pipe)

def scale_svd_rf_pipe_new_import():
    if False:
        i = 10
        return i + 15
    from h2o.estimators.svd import H2OSingularValueDecompositionEstimator
    print('Importing USArrests.csv data...')
    arrests = h2o.upload_file(pyunit_utils.locate('smalldata/pca_test/USArrests.csv'))
    print('Compare with SVD')
    try:
        pipe = Pipeline([('standardize', H2OScaler()), ('svd', H2OSingularValueDecompositionEstimator(nv=3)), ('rf', H2ORandomForestEstimator(seed=42, ntrees=50))])
        pipe.fit(arrests[1:], arrests[0])
        assert False, 'Pipeline should fail without using H2OSingularValueDecompositionEstimator.init_for_pipeline()'
    except TypeError:
        pass
    pipe = Pipeline([('standardize', H2OScaler()), ('svd', H2OSingularValueDecompositionEstimator(nv=3).init_for_pipeline()), ('rf', H2ORandomForestEstimator(seed=42, ntrees=50))])
    pipe.fit(arrests[1:], arrests[0])
    print(pipe)
if __name__ == '__main__':
    pyunit_utils.standalone_test(scale_svd_rf_pipe)
    pyunit_utils.standalone_test(scale_svd_rf_pipe_new_import)
else:
    scale_svd_rf_pipe()
    scale_svd_rf_pipe_new_import()