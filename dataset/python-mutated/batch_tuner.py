"""
batch_tuner.py including:
    class BatchTuner
"""
import logging
import nni
from nni.common.hpo_utils import validate_search_space
from nni.tuner import Tuner
TYPE = '_type'
CHOICE = 'choice'
VALUE = '_value'
LOGGER = logging.getLogger('batch_tuner_AutoML')

class BatchTuner(Tuner):
    """
    Batch tuner is a special tuner that allows users to simply provide several hyperparameter sets,
    and it will evaluate each set.

    Batch tuner does **not** support standard search space.

    Search space of batch tuner looks like a single ``choice`` in standard search space,
    but it has different meaning.

    Consider following search space:

    .. code-block::

        'combine_params': {
            '_type': 'choice',
            '_value': [
                {'x': 0, 'y': 1},
                {'x': 1, 'y': 2},
                {'x': 1, 'y': 3},
            ]
        }

    Batch tuner will generate following 4 hyperparameter sets:

    1. {'x': 0, 'y': 1}
    2. {'x': 1, 'y': 2}
    3. {'x': 1, 'y': 3}

    If this search space was used with grid search tuner, it would instead generate:

    1. {'combine_params': {'x': 0, 'y': 1 }}
    2. {'combine_params': {'x': 1, 'y': 2 }}
    3. {'combine_params': {'x': 1, 'y': 3 }}

    Examples
    --------

    .. code-block::

        config.search_space = {
            'combine_params': {
                '_type': 'choice',
                '_value': [
                    {'optimizer': 'Adam', 'learning_rate': 0.001},
                    {'optimizer': 'Adam', 'learning_rate': 0.0001},
                    {'optimizer': 'Adam', 'learning_rate': 0.00001},
                    {'optimizer': 'SGD', 'learning_rate': 0.01},
                    {'optimizer': 'SGD', 'learning_rate': 0.005},
                ]
            }
        }
        config.tuner.name = 'Batch'
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._count = -1
        self._values = []

    def _is_valid(self, search_space):
        if False:
            i = 10
            return i + 15
        "\n        Check the search space is valid: only contains 'choice' type\n\n        Parameters\n        ----------\n        search_space : dict\n\n        Returns\n        -------\n        None or list\n            If valid, return candidate values; else return None.\n        "
        if not len(search_space) == 1:
            raise RuntimeError('BatchTuner only supprt one combined-paramreters key.')
        for param in search_space:
            param_type = search_space[param][TYPE]
            if not param_type == CHOICE:
                raise RuntimeError('BatchTuner only supprt                                     one combined-paramreters type is choice.')
            if isinstance(search_space[param][VALUE], list):
                return search_space[param][VALUE]
            raise RuntimeError('The combined-paramreters                                 value in BatchTuner is not a list.')
        return None

    def update_search_space(self, search_space):
        if False:
            for i in range(10):
                print('nop')
        validate_search_space(search_space, ['choice'])
        self._values = self._is_valid(search_space)

    def generate_parameters(self, parameter_id, **kwargs):
        if False:
            print('Hello World!')
        self._count += 1
        if self._count > len(self._values) - 1:
            raise nni.NoMoreTrialError('no more parameters now.')
        return self._values[self._count]

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        if False:
            i = 10
            return i + 15
        pass

    def import_data(self, data):
        if False:
            i = 10
            return i + 15
        if not self._values:
            LOGGER.info('Search space has not been initialized, skip this data import')
            return
        self._values = self._values[self._count + 1:]
        self._count = -1
        _completed_num = 0
        for trial_info in data:
            LOGGER.info('Importing data, current processing                             progress %s / %s', _completed_num, len(data))
            assert 'parameter' in trial_info
            _params = trial_info['parameter']
            assert 'value' in trial_info
            _value = trial_info['value']
            if not _value:
                LOGGER.info('Useless trial data, value is %s, skip this trial data.', _value)
                continue
            _completed_num += 1
            if _params in self._values:
                self._values.remove(_params)
        LOGGER.info('Successfully import data to batch tuner,                         total data: %d, imported data: %d.', len(data), _completed_num)