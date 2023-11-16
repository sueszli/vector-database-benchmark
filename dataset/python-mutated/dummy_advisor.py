import logging
from collections import defaultdict
import json_tricks
import numpy as np
from nni import parameter_expressions as param
from nni.msg_dispatcher_base import MsgDispatcherBase
from nni.protocol import CommandType, send
from nni.utils import MetricType
logger = logging.getLogger('customized_advisor')

class DummyAdvisor(MsgDispatcherBase):
    """WARNING: Advisor API is subject to change in future releases.

    This advisor creates a new trial when validation accuracy of any one of the trials just dropped.
    The trial is killed if the validation accuracy doesn't improve for at least k last-reported metrics.
    To demonstrate the high flexibility of writing advisors, we don't use tuners or the standard definition of
    search space. This is just a demo to customize an advisor. It's not intended to make any sense.
    """

    def __init__(self, k=3):
        if False:
            return 10
        super(DummyAdvisor, self).__init__()
        self.k = k
        self.random_state = np.random.RandomState()

    def handle_initialize(self, data):
        if False:
            for i in range(10):
                print('nop')
        logger.info('Advisor initialized: {}'.format(data))
        self.handle_update_search_space(data)
        self.parameters_count = 0
        self.parameter_best_metric = defaultdict(float)
        self.parameter_cooldown = defaultdict(int)
        send(CommandType.Initialized, '')

    def _send_new_trial(self):
        if False:
            while True:
                i = 10
        self.parameters_count += 1
        new_trial = {'parameter_id': self.parameters_count, 'parameters': {'optimizer': param.choice(self.searchspace_json['optimizer'], self.random_state), 'learning_rate': param.loguniform(self.searchspace_json['learning_rate'][0], self.searchspace_json['learning_rate'][1], self.random_state)}, 'parameter_source': 'algorithm'}
        logger.info('New trial sent: {}'.format(new_trial))
        send(CommandType.NewTrialJob, json_tricks.dumps(new_trial))

    def handle_request_trial_jobs(self, data):
        if False:
            return 10
        logger.info('Request trial jobs: {}'.format(data))
        for _ in range(data):
            self._send_new_trial()

    def handle_update_search_space(self, data):
        if False:
            return 10
        logger.info('Search space update: {}'.format(data))
        self.searchspace_json = data

    def handle_trial_end(self, data):
        if False:
            i = 10
            return i + 15
        logger.info('Trial end: {}'.format(data))

    def handle_report_metric_data(self, data):
        if False:
            print('Hello World!')
        logger.info('Metric reported: {}'.format(data))
        if data['type'] == MetricType.REQUEST_PARAMETER:
            raise ValueError('Request parameter not supported')
        elif data['type'] == MetricType.PERIODICAL:
            parameter_id = data['parameter_id']
            if data['value'] > self.parameter_best_metric[parameter_id]:
                self.parameter_best_metric[parameter_id] = data['value']
                self.parameter_cooldown[parameter_id] = 0
            else:
                self.parameter_cooldown[parameter_id] += 1
                logger.info('Accuracy dropped, cooldown {}, sending a new trial'.format(self.parameter_cooldown[parameter_id]))
                self._send_new_trial()
                if self.parameter_cooldown[parameter_id] >= self.k:
                    logger.info('Send kill signal to {}'.format(data))
                    send(CommandType.KillTrialJob, json_tricks.dumps(data['trial_job_id']))