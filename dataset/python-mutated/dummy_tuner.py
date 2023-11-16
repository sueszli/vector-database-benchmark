from nni.tuner import Tuner

class DummyTuner(Tuner):

    def generate_parameters(self, parameter_id):
        if False:
            i = 10
            return i + 15
        return 'unit-test-parm'

    def generate_multiple_parameters(self, parameter_id_list):
        if False:
            i = 10
            return i + 15
        return ['unit-test-param1', 'unit-test-param2']

    def receive_trial_result(self, parameter_id, parameters, value):
        if False:
            for i in range(10):
                print('nop')
        pass

    def receive_customized_trial_result(self, parameter_id, parameters, value):
        if False:
            return 10
        pass

    def update_search_space(self, search_space):
        if False:
            for i in range(10):
                print('nop')
        pass