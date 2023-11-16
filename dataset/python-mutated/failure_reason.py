from abc import abstractmethod
from typing import NamedTuple
from trashcli.compat import Protocol
from trashcli.lib.environ import Environ
from trashcli.put.core.candidate import Candidate

class LogContext(NamedTuple('LogContext', [('trashee_path', str), ('candidate', Candidate), ('environ', Environ)])):

    def shrunk_candidate_path(self):
        if False:
            while True:
                i = 10
        return self.candidate.shrink_user(self.environ)

    def trash_dir_norm_path(self):
        if False:
            print('Hello World!')
        return self.candidate.norm_path()

    def files_dir(self):
        if False:
            while True:
                i = 10
        return self.candidate.files_dir()

class FailureReason(Protocol):

    @abstractmethod
    def log_entries(self, context):
        if False:
            while True:
                i = 10
        raise NotImplementedError