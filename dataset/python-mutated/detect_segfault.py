from metaflow_test import MetaflowTest, ExpectationFailed, steps

class DetectSegFaultTest(MetaflowTest):
    """
    Test that segmentation faults produce a message in the logs
    """
    PRIORITY = 2
    SHOULD_FAIL = True

    @steps(0, ['singleton-end'], required=True)
    def step_end(self):
        if False:
            while True:
                i = 10
        import ctypes
        print('Crash and burn!')
        ctypes.string_at(0)

    @steps(1, ['all'])
    def step_all(self):
        if False:
            return 10
        pass

    def check_results(self, flow, checker):
        if False:
            i = 10
            return i + 15
        run = checker.get_run()
        if run:
            checker.assert_log('end', 'stdout', 'Crash and burn!', exact_match=False)
            checker.assert_log('end', 'stderr', 'segmentation fault', exact_match=False)