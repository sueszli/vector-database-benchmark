from __future__ import annotations
import logging
from typing_extensions import Literal
import nni
from nni.runtime.command_channel.http import HttpChannel
from nni.typehint import ParameterRecord, TrialMetric
from .base import TrialCommandChannel
_logger = logging.getLogger(__name__)

class TrialCommandChannelV3(TrialCommandChannel):

    def __init__(self, url: str):
        if False:
            print('Hello World!')
        assert url.startswith('http://'), 'Only support HTTP command channel'
        _logger.info(f'Connect to trial command channel {url}')
        self._channel: HttpChannel = HttpChannel(url)

    def receive_parameter(self) -> ParameterRecord | None:
        if False:
            print('Hello World!')
        req = {'type': 'request_parameter'}
        self._channel.send(req)
        res = self._channel.receive()
        if res is None:
            _logger.error('Trial command channel is closed')
            return None
        assert res['type'] == 'parameter'
        return nni.load(res['parameter'])

    def send_metric(self, type: Literal['PERIODICAL', 'FINAL'], parameter_id: int | None, trial_job_id: str, sequence: int, value: TrialMetric) -> None:
        if False:
            print('Hello World!')
        metric = {'parameter_id': parameter_id, 'trial_job_id': trial_job_id, 'type': type, 'sequence': sequence, 'value': nni.dump(value)}
        command = {'type': 'metric', 'metric': nni.dump(metric)}
        self._channel.send(command)