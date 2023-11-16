from builtins import zip
from builtins import range
from past.utils import old_div
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.kmeans import H2OKMeansEstimator

def convergeKmeans():
    if False:
        i = 10
        return i + 15
    ozone_h2o = h2o.import_file(path=pyunit_utils.locate('smalldata/glm_test/ozone.csv'))
    miters = 5
    ncent = 10
    start = ozone_h2o[0:10, 0:4]
    try:
        H2OKMeansEstimator(max_iterations=0).train(x=list(range(ozone_h2o.ncol)), training_frame=ozone_h2o)
        assert False, 'expected an error'
    except EnvironmentError:
        assert True
    centers = start
    for i in range(miters):
        rep_fit = H2OKMeansEstimator(k=ncent, user_points=centers, max_iterations=1)
        rep_fit.train(x=list(range(ozone_h2o.ncol)), training_frame=ozone_h2o)
        centers = h2o.H2OFrame(rep_fit.centers())
    all_fit = H2OKMeansEstimator(k=ncent, user_points=start, max_iterations=miters)
    all_fit.train(x=list(range(ozone_h2o.ncol)), training_frame=ozone_h2o)
    assert rep_fit.centers() == all_fit.centers(), 'expected the centers to be the same'
    all_fit2 = H2OKMeansEstimator(k=ncent, user_points=h2o.H2OFrame(all_fit.centers()), max_iterations=1)
    all_fit2.train(x=list(range(ozone_h2o.ncol)), training_frame=ozone_h2o)
    avg_change = old_div(sum([sum([pow(e1 - e2, 2) for (e1, e2) in zip(c1, c2)]) for (c1, c2) in zip(all_fit.centers(), all_fit2.centers())]), ncent)
    assert avg_change < 1e-06 or all_fit._model_json['output']['iterations'] == miters
if __name__ == '__main__':
    pyunit_utils.standalone_test(convergeKmeans)
else:
    convergeKmeans()