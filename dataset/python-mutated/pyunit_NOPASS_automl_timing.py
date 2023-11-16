import sys, os, time
sys.path.insert(1, os.path.join('..', '..', '..'))
from tests import pyunit_utils as pu
from h2o.automl import H2OAutoML
from _automl_utils import import_dataset
"\nThose tests check time constraints on AutoML runs and can be fragile when run on Jenkins, \nhence the NOPASS prefix that won't fail the build if they don't pass.\n"

def test_automl_stops_after_max_runtime_secs():
    if False:
        i = 10
        return i + 15
    print('Check that automl gets interrupted after `max_runtime_secs`')
    max_runtime_secs = 30
    cancel_tolerance_secs = 5 + 5
    ds = import_dataset()
    aml = H2OAutoML(project_name='py_aml_max_runtime_secs', seed=1, max_runtime_secs=max_runtime_secs)
    start = time.time()
    aml.train(y=ds.target, training_frame=ds.train)
    end = time.time()
    assert abs(end - start - max_runtime_secs) < cancel_tolerance_secs, end - start

def test_no_model_takes_more_than_max_runtime_secs_per_model():
    if False:
        i = 10
        return i + 15
    print('Check that individual model get interrupted after `max_runtime_secs_per_model`')
    ds = import_dataset(seed=1, larger=True)
    max_runtime_secs = 30
    models_count = {}
    for max_runtime_secs_per_model in [0, 3, max_runtime_secs]:
        aml = H2OAutoML(project_name='py_aml_max_runtime_secs_per_model_{}'.format(max_runtime_secs_per_model), seed=1, max_runtime_secs_per_model=max_runtime_secs_per_model, max_runtime_secs=max_runtime_secs)
        aml.train(y=ds.target, training_frame=ds.train)
        models_count[max_runtime_secs_per_model] = len(aml.leaderboard)
    assert abs(models_count[0] - models_count[max_runtime_secs]) <= 1
    assert abs(models_count[0] - models_count[3]) > 1
pu.run_tests([test_automl_stops_after_max_runtime_secs, test_no_model_takes_more_than_max_runtime_secs_per_model])