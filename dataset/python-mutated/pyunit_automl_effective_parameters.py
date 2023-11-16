import sys, os
import random
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
from h2o.automl import H2OAutoML
from tests import pyunit_utils as pu
from _automl_utils import get_partitioned_model_names
if sys.version_info[0] < 3:
    automl_seed = random.randint(0, sys.maxint)
else:
    automl_seed = random.randint(0, sys.maxsize)
print('Random Seed for pyunit_automl_leaderboard.py = ' + str(automl_seed))

def check_model_property(model_names, prop_name, present=True, actual_value=None, default_value=None, input_value=None):
    if False:
        for i in range(10):
            print('nop')
    for mn in model_names:
        model = h2o.get_model(mn)
        if present:
            assert prop_name in model.params.keys(), 'missing {prop} in model {model}'.format(prop=prop_name, model=mn)
            assert actual_value is None or model.params[prop_name]['actual'] == actual_value, 'actual value for {prop} in model {model} is {val}, expected {exp}'.format(prop=prop_name, model=mn, val=model.params[prop_name]['actual'], exp=actual_value)
            assert default_value is None or model.params[prop_name]['default'] == default_value, 'default value for {prop} in model {model} is {val}, expected {exp}'.format(prop=prop_name, model=mn, val=model.params[prop_name]['default'], exp=default_value)
            assert input_value is None or model.params[prop_name]['input'] == input_value, 'default value for {prop} in model {model} is {val}, expected {exp}'.format(prop=prop_name, model=mn, val=model.params[prop_name]['input'], exp=input_value)
        else:
            assert prop_name not in model.params.keys(), 'unexpected {prop} in model {model}'.format(prop=prop_name, model=mn)

def test_actual_default_input_stopping_rounds():
    if False:
        return 10
    train = h2o.import_file(path=pu.locate('smalldata/extdata/australia.csv'))
    target = 'runoffnew'
    exclude_algos = ['DeepLearning', 'GLM']
    aml = H2OAutoML(project_name='actual_default_input_stopping_rounds', exclude_algos=exclude_algos, max_models=10, seed=automl_seed)
    aml.train(y=target, training_frame=train)
    base_models = get_partitioned_model_names(aml.leaderboard).base
    check_model_property(base_models, 'stopping_rounds', True, 0, 0, 3)
pu.run_tests([test_actual_default_input_stopping_rounds])