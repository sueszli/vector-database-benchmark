from abc import ABC, abstractmethod
import schemas

class BaseCollaboration(ABC):

    @classmethod
    @abstractmethod
    def add(cls, tenant_id, data: schemas.AddCollaborationSchema):
        if False:
            return 10
        pass

    @classmethod
    @abstractmethod
    def say_hello(cls, url):
        if False:
            print('Hello World!')
        pass

    @classmethod
    @abstractmethod
    def send_raw(cls, tenant_id, webhook_id, body):
        if False:
            i = 10
            return i + 15
        pass

    @classmethod
    @abstractmethod
    def send_batch(cls, tenant_id, webhook_id, attachments):
        if False:
            return 10
        pass

    @classmethod
    @abstractmethod
    def __share(cls, tenant_id, integration_id, attachments):
        if False:
            print('Hello World!')
        pass

    @classmethod
    @abstractmethod
    def share_session(cls, tenant_id, project_id, session_id, user, comment, integration_id=None):
        if False:
            return 10
        pass

    @classmethod
    @abstractmethod
    def share_error(cls, tenant_id, project_id, error_id, user, comment, integration_id=None):
        if False:
            while True:
                i = 10
        pass

    @classmethod
    @abstractmethod
    def get_integration(cls, tenant_id, integration_id=None):
        if False:
            for i in range(10):
                print('nop')
        pass