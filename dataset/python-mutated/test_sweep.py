import pytest
from recommenders.tuning.parameter_sweep import generate_param_grid

@pytest.fixture(scope='module')
def parameter_dictionary():
    if False:
        for i in range(10):
            print('nop')
    return {'param1': [1, 2, 3], 'param2': [4, 5, 6], 'param3': 1}

def test_param_sweep(parameter_dictionary):
    if False:
        i = 10
        return i + 15
    params_grid = generate_param_grid(parameter_dictionary)
    assert params_grid == [{'param1': 1, 'param2': 4, 'param3': 1}, {'param1': 1, 'param2': 5, 'param3': 1}, {'param1': 1, 'param2': 6, 'param3': 1}, {'param1': 2, 'param2': 4, 'param3': 1}, {'param1': 2, 'param2': 5, 'param3': 1}, {'param1': 2, 'param2': 6, 'param3': 1}, {'param1': 3, 'param2': 4, 'param3': 1}, {'param1': 3, 'param2': 5, 'param3': 1}, {'param1': 3, 'param2': 6, 'param3': 1}]