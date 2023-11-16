import pytest
try:
    from ray import tune
    from ludwig.hyperopt.execution import get_build_hyperopt_executor
except ImportError:
    RAY_AVAILABLE = False
else:
    RAY_AVAILABLE = True
from ludwig.constants import RAY, TYPE
HYPEROPT_PARAMS = {'test_1': {'parameters': {'trainer.learning_rate': {'space': 'uniform', 'lower': 0.001, 'upper': 0.1}, 'combiner.num_fc_layers': {'space': 'qrandint', 'lower': 3, 'upper': 6, 'q': 3}, 'utterance.cell_type': {'space': 'grid_search', 'values': ['rnn', 'gru', 'lstm']}}}, 'test_2': {'parameters': {'trainer.learning_rate': {'space': 'loguniform', 'lower': 0.001, 'upper': 0.1, 'base': 10}, 'combiner.num_fc_layers': {'space': 'randint', 'lower': 2, 'upper': 6}, 'utterance.cell_type': {'space': 'choice', 'categories': ['rnn', 'gru', 'lstm']}}}}
if RAY_AVAILABLE:
    EXPECTED_SEARCH_SPACE = {'test_1': {'trainer.learning_rate': tune.uniform(0.001, 0.1), 'combiner.num_fc_layers': tune.qrandint(3, 6, 3), 'utterance.cell_type': tune.grid_search(['rnn', 'gru', 'lstm'])}, 'test_2': {'trainer.learning_rate': tune.loguniform(0.001, 0.1), 'combiner.num_fc_layers': tune.randint(2, 6), 'utterance.cell_type': tune.choice(['rnn', 'gru', 'lstm'])}}

@pytest.mark.skipif(not RAY_AVAILABLE, reason='Ray is not installed for testing')
@pytest.mark.parametrize('key', ['test_1', 'test_2'])
def test_grid_strategy(key):
    if False:
        print('Hello World!')
    hyperopt_test_params = HYPEROPT_PARAMS[key]
    expected_search_space = EXPECTED_SEARCH_SPACE[key]
    tune_sampler_params = hyperopt_test_params['parameters']
    hyperopt_executor = get_build_hyperopt_executor(RAY)(tune_sampler_params, 'output_feature', 'mse', 'minimize', 'validation', search_alg={TYPE: 'variant_generator'}, **{'type': 'ray', 'num_samples': 2, 'scheduler': {'type': 'fifo'}})
    search_space = hyperopt_executor.search_space
    actual_params_keys = search_space.keys()
    expected_params_keys = expected_search_space.keys()
    for param in search_space:
        assert isinstance(search_space[param], type(expected_search_space[param]))
    assert actual_params_keys == expected_params_keys