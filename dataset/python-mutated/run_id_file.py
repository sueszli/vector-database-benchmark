from metaflow_test import MetaflowTest, ExpectationFailed, steps

class RunIdFileTest(MetaflowTest):
    """
    Resuming and initial running of a flow should write run id file early (prior to execution)
    """
    RESUME = True
    PRIORITY = 3

    @steps(0, ['singleton-start'], required=True)
    def step_start(self):
        if False:
            i = 10
            return i + 15
        import os
        from metaflow import current
        assert os.path.isfile('run-id'), 'run id file should exist before resume execution'
        with open('run-id', 'r') as f:
            run_id_from_file = f.read()
        assert run_id_from_file == current.run_id
        if not is_resumed():
            raise ResumeFromHere()

    @steps(2, ['all'])
    def step_all(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def check_results(self, flow, checker):
        if False:
            while True:
                i = 10
        pass