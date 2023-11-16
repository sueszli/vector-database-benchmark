from metaflow_test import MetaflowTest, ExpectationFailed, steps, tag

class CardTimeoutTest(MetaflowTest):
    """
    Test that checks if the card decorator works as intended with the timeout decorator.
    # This test set an artifact in the steps and also set a timeout to the card argument.
    # It will assert the artifact to be None.
    """
    PRIORITY = 2

    @tag('card(type="test_timeout_card",timeout=10,options=dict(timeout=20),save_errors=False)')
    @steps(0, ['start'])
    def step_start(self):
        if False:
            for i in range(10):
                print('nop')
        from metaflow import current
        self.task = current.pathspec

    @steps(1, ['all'])
    def step_all(self):
        if False:
            print('Hello World!')
        pass

    def check_results(self, flow, checker):
        if False:
            return 10
        run = checker.get_run()
        for step in flow:
            if step.name != 'start':
                continue
            if run is None:
                cli_check_dict = checker.artifact_dict(step.name, 'task')
                for task_pathspec in cli_check_dict:
                    task_id = task_pathspec.split('/')[-1]
                    checker.assert_card(step.name, task_id, 'timeout_card', None)
            else:
                meta_check_dict = checker.artifact_dict(step.name, 'task')
                for task_id in meta_check_dict:
                    checker.assert_card(step.name, task_id, 'timeout_card', None)