CUSTOM_HEADER_NAME = 'X-SOME-HEADER'
from dagster import DagsterRun, QueuedRunCoordinator, SubmitRunContext

class CustomRunCoordinator(QueuedRunCoordinator):

    def submit_run(self, context: SubmitRunContext) -> DagsterRun:
        if False:
            print('Hello World!')
        desired_header = context.get_request_header(CUSTOM_HEADER_NAME)
        ...