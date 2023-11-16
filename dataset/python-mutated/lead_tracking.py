import abc
import typing
from users.models import FFAdminUser

class LeadTracker(abc.ABC):

    def __init__(self, client: typing.Any=None):
        if False:
            while True:
                i = 10
        self.client = client or self._get_client()

    @staticmethod
    @abc.abstractmethod
    def should_track(user: FFAdminUser) -> bool:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @abc.abstractmethod
    def create_lead(self, user: FFAdminUser):
        if False:
            while True:
                i = 10
        pass

    @abc.abstractmethod
    def _get_client(self) -> typing.Any:
        if False:
            return 10
        pass