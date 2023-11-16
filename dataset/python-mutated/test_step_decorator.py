from metaflow.decorators import StepDecorator

class TestStepDecorator(StepDecorator):
    name = 'test_step_decorator'

    def task_post_step(self, step_name, flow, graph, retry_count, max_user_code_retries):
        if False:
            while True:
                i = 10
        flow.plugin_set_value = step_name