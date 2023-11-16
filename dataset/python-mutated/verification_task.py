import typing
from twisted.internet.defer import Deferred, succeed
from golem.core.common import deadline_to_timeout

class VerificationTask:

    def __init__(self, subtask_id, deadline, kwargs) -> None:
        if False:
            i = 10
            return i + 15
        self.deadline = deadline
        self.kwargs = kwargs
        self.subtask_id = subtask_id
        self.verifier: typing.Any = None

    def start(self, verifier_class) -> Deferred:
        if False:
            i = 10
            return i + 15
        self.verifier = verifier_class(self.kwargs)
        if deadline_to_timeout(self.deadline) > 0:
            if self.verifier.simple_verification():
                return self.verifier.start_verification()
            return succeed(self.verifier.verification_completed())
        else:
            return succeed(self.verifier.task_timeout(self.subtask_id))

    def get_results(self):
        if False:
            i = 10
            return i + 15
        return self.verifier.verification_completed()

    def stop(self, finished):
        if False:
            while True:
                i = 10
        self.verifier.stop()
        finished.cancel()