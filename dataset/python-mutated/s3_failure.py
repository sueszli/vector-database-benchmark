from metaflow_test import MetaflowTest, ExpectationFailed, steps

class S3FailureTest(MetaflowTest):
    """
    Test that S3 failures are handled correctly.
    """
    PRIORITY = 1
    HEADER = "\nimport os\n\nos.environ['TEST_S3_RETRY'] = '1'\n"

    @steps(0, ['singleton-start'], required=True)
    def step_start(self):
        if False:
            for i in range(10):
                print('nop')
        from metaflow import current
        self.x = '%s/%s' % (current.flow_name, current.run_id)

    @steps(0, ['end'])
    def step_end(self):
        if False:
            i = 10
            return i + 15
        from metaflow import current
        run_id = '%s/%s' % (current.flow_name, current.run_id)
        assert_equals(self.x, run_id)

    @steps(1, ['all'])
    def step_all(self):
        if False:
            i = 10
            return i + 15
        pass

    def check_results(self, flow, checker):
        if False:
            i = 10
            return i + 15
        run = checker.get_run()
        if run:
            checker.assert_log('start', 'stderr', 'TEST_S3_RETRY', exact_match=False)
        run_id = 'S3FailureTestFlow/%s' % checker.run_id
        checker.assert_artifact('start', 'x', run_id)
        checker.assert_artifact('end', 'x', run_id)