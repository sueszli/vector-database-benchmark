from metaflow_test import MetaflowTest, ExpectationFailed, steps

class TaskExceptionTest(MetaflowTest):
    """
    A test to validate if exceptions are stored and retrieved correctly
    """
    PRIORITY = 1
    SHOULD_FAIL = True

    @steps(0, ['singleton-end'], required=True)
    def step_start(self):
        if False:
            return 10
        raise KeyError('Something has gone wrong')

    @steps(2, ['all'])
    def step_all(self):
        if False:
            print('Hello World!')
        pass

    def check_results(self, flow, checker):
        if False:
            for i in range(10):
                print('nop')
        run = checker.get_run()
        if run is not None:
            for task in run['end']:
                assert_equals('KeyError' in str(task.exception), True)
                assert_equals(task.exception.exception, "'Something has gone wrong'")