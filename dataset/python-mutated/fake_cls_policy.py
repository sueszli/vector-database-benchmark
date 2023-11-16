from ding.policy import Policy
from ding.model import model_wrap

class fake_policy(Policy):

    def _init_learn(self):
        if False:
            print('Hello World!')
        pass

    def _forward_learn(self, data):
        if False:
            i = 10
            return i + 15
        pass

    def _init_eval(self):
        if False:
            while True:
                i = 10
        self._eval_model = model_wrap(self._model, 'base')

    def _forward_eval(self, data):
        if False:
            print('Hello World!')
        self._eval_model.eval()
        output = self._eval_model.forward(data)
        return output

    def _monitor_vars_learn(self):
        if False:
            return 10
        return ['forward_time', 'backward_time', 'sync_time']

    def _init_collect(self):
        if False:
            while True:
                i = 10
        pass

    def _forward_collect(self, data):
        if False:
            print('Hello World!')
        pass

    def _process_transition(self):
        if False:
            while True:
                i = 10
        pass

    def _get_train_sample(self):
        if False:
            return 10
        pass