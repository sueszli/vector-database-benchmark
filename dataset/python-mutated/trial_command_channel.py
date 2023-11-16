from __future__ import annotations
import copy
from typing_extensions import Literal
from nni.runtime.trial_command_channel import TrialCommandChannel
from nni.typehint import TrialMetric, ParameterRecord

class TestHelperTrialCommandChannel(TrialCommandChannel):

    def __init__(self):
        if False:
            return 10
        self._params = {'parameter_id': 0, 'parameters': {}}
        self._last_metric = None
        self.intermediates = []
        self.final = None

    def init_params(self, params):
        if False:
            print('Hello World!')
        self._params = copy.deepcopy(params)

    def get_last_metric(self):
        if False:
            for i in range(10):
                print('nop')
        'For backward compatibility, return the last metric as the full dict.'
        return self._last_metric

    def receive_parameter(self) -> ParameterRecord | None:
        if False:
            while True:
                i = 10
        return self._params

    def send_metric(self, type: Literal['PERIODICAL', 'FINAL'], parameter_id: int | None, trial_job_id: str, sequence: int, value: TrialMetric) -> None:
        if False:
            while True:
                i = 10
        self._last_metric = {'type': type, 'parameter_id': parameter_id, 'trial_job_id': trial_job_id, 'sequence': sequence, 'value': value}
        if type == 'PERIODICAL':
            self.intermediates.append(value)
        else:
            self.final = value