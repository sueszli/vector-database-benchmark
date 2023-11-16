from builtins import range
import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def test_relevel_by_freq_glm():
    if False:
        print('Hello World!')
    prostate_cat = h2o.import_file(path=pyunit_utils.locate('smalldata/prostate/prostate_cat.csv'))
    dpros_levels = prostate_cat['DPROS'].levels()[0]
    assert dpros_levels == ['Both', 'Left', 'None', 'Right']
    dpros_levels_ordered = prostate_cat['DPROS'].table().as_data_frame().sort_values(by='Count')['DPROS'].tolist()
    assert dpros_levels_ordered == ['Both', 'Right', 'None', 'Left']
    mh2o1 = H2OGeneralizedLinearEstimator(family='binomial', lambda_=0, missing_values_handling='Skip')
    mh2o1.train(x=list(range(1, prostate_cat.ncol)), y=0, training_frame=prostate_cat)
    ns = mh2o1.coef().keys()
    print(ns)
    assert 'DPROS.Left' in ns, 'Left level IS NOT expected to be skipped by default'
    assert 'DPROS.Both' not in ns, 'Both level IS expected to be skipped by default'
    prostate_cat_relevel = prostate_cat.relevel_by_frequency()
    dpros_relevel_levels = prostate_cat_relevel['DPROS'].table().as_data_frame()['DPROS'].tolist()
    assert dpros_relevel_levels == ['Left', 'None', 'Right', 'Both']
    mh2o2 = H2OGeneralizedLinearEstimator(family='binomial', lambda_=0, missing_values_handling='Skip')
    mh2o2.train(x=list(range(1, prostate_cat_relevel.ncol)), y=0, training_frame=prostate_cat_relevel)
    ns2 = mh2o2.coef().keys()
    print(ns2)
    assert 'DPROS.Left' not in ns2, 'Left level IS expected to be skipped by default'
    assert 'DPROS.Both' in ns2, 'Both level IS NOT expected to be skipped by default'
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_relevel_by_freq_glm)
else:
    test_relevel_by_freq_glm()