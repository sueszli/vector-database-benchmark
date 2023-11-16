from metaflow_test import MetaflowTest, ExpectationFailed, steps

class ResumeStartStepTest(MetaflowTest):
    """
    Resuming from the start step should work
    """
    RESUME = True
    PRIORITY = 3
    PARAMETERS = {'int_param': {'default': 123}}

    @steps(0, ['singleton-start'], required=True)
    def step_start(self):
        if False:
            print('Hello World!')
        from metaflow import current
        if is_resumed():
            self.data = 'foo'
            self.actual_origin_run_id = current.origin_run_id
            from metaflow_test import origin_run_id_for_resume
            self.expected_origin_run_id = origin_run_id_for_resume()
            assert len(self.expected_origin_run_id) > 0
        else:
            self.data = 'bar'
            raise ResumeFromHere()

    @steps(2, ['all'])
    def step_all(self):
        if False:
            while True:
                i = 10
        pass

    def check_results(self, flow, checker):
        if False:
            i = 10
            return i + 15
        run = checker.get_run()
        if run is None:
            for step in flow:
                checker.assert_artifact(step.name, 'data', 'foo')
                checker.assert_artifact(step.name, 'int_param', 123)
        else:
            assert_equals(run.data.expected_origin_run_id, run.data.actual_origin_run_id)
            exclude_keys = ['origin-task-id', 'origin-run-id']
            resumed_metadata = run['start'].task.metadata_dict
            assert 'origin-task-id' not in resumed_metadata, 'Invalid clone'
            assert 'origin-run-id' in resumed_metadata, 'Invalid resume'