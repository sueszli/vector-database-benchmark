import logging
import time
from nni.tuner import Tuner

class MultiThreadTuner(Tuner):

    def __init__(self):
        if False:
            return 10
        self.parent_done = False

    def generate_parameters(self, parameter_id, **kwargs):
        if False:
            while True:
                i = 10
        logging.debug('generate_parameters: %s %s', parameter_id, kwargs)
        if parameter_id == 0:
            return {'x': 0}
        else:
            while not self.parent_done:
                logging.debug('parameter_id %s sleeping', parameter_id)
                time.sleep(2)
            logging.debug('parameter_id %s waked up', parameter_id)
            return {'x': 1}

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        if False:
            i = 10
            return i + 15
        logging.debug('receive_trial_result: %s %s %s %s', parameter_id, parameters, value, kwargs)
        if parameter_id == 0:
            self.parent_done = True

    def update_search_space(self, search_space):
        if False:
            print('Hello World!')
        pass