from datetime import datetime
from robot.errors import BreakLoop, ContinueLoop, DataError, ExecutionFailed, ExecutionStatus, HandlerExecutionFailed, ReturnFromKeyword
from robot.utils import ErrorDetails

class StatusReporter:

    def __init__(self, data, result, context, run=True, suppress=False):
        if False:
            while True:
                i = 10
        self.data = data
        self.result = result
        self.context = context
        if run:
            self.pass_status = result.PASS
            result.status = result.NOT_SET
        else:
            self.pass_status = result.status = result.NOT_RUN
        self.suppress = suppress
        self.initial_test_status = None

    def __enter__(self):
        if False:
            print('Hello World!')
        context = self.context
        result = self.result
        self.initial_test_status = context.test.status if context.test else None
        if not result.start_time:
            result.start_time = datetime.now()
        context.start_body_item(self.data, result)
        if result.type in result.KEYWORD_TYPES:
            self._warn_if_deprecated(result.doc, result.full_name)
        return self

    def _warn_if_deprecated(self, doc, name):
        if False:
            i = 10
            return i + 15
        if doc.startswith('*DEPRECATED') and '*' in doc[1:]:
            message = ' ' + doc.split('*', 2)[-1].strip()
            self.context.warn(f"Keyword '{name}' is deprecated.{message}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        context = self.context
        result = self.result
        failure = self._get_failure(exc_type, exc_val, exc_tb, context)
        if failure is None:
            result.status = self.pass_status
        else:
            result.status = failure.status
            if not isinstance(failure, (BreakLoop, ContinueLoop, ReturnFromKeyword)):
                result.message = failure.message
        if self.initial_test_status == 'PASS':
            context.test.status = result.status
        result.elapsed_time = datetime.now() - result.start_time
        context.end_body_item(self.data, result)
        if failure is not exc_val and (not self.suppress):
            raise failure
        return self.suppress

    def _get_failure(self, exc_type, exc_value, exc_tb, context):
        if False:
            for i in range(10):
                print('nop')
        if exc_value is None:
            return None
        if isinstance(exc_value, ExecutionStatus):
            return exc_value
        if isinstance(exc_value, DataError):
            msg = exc_value.message
            context.fail(msg)
            return ExecutionFailed(msg, syntax=exc_value.syntax)
        error = ErrorDetails(exc_value)
        failure = HandlerExecutionFailed(error)
        if failure.timeout:
            context.timeout_occurred = True
        if failure.skip:
            context.skip(error.message)
        else:
            context.fail(error.message)
        if error.traceback:
            context.debug(error.traceback)
        return failure