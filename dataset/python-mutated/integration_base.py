from abc import ABC, abstractmethod
from chalicelib.utils import pg_client, helper

class BaseIntegration(ABC):

    def __init__(self, user_id, ISSUE_CLASS):
        if False:
            return 10
        self._user_id = user_id
        self._issue_handler = ISSUE_CLASS(self.integration_token)

    @property
    @abstractmethod
    def provider(self):
        if False:
            return 10
        pass

    @property
    @abstractmethod
    def issue_handler(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    def integration_token(self):
        if False:
            print('Hello World!')
        integration = self.get()
        if integration is None:
            print('no token configured yet')
            return None
        return integration['token']

    def get(self):
        if False:
            print('Hello World!')
        with pg_client.PostgresClient() as cur:
            cur.execute(cur.mogrify('SELECT *\n                        FROM public.oauth_authentication \n                        WHERE user_id=%(user_id)s AND provider=%(provider)s;', {'user_id': self._user_id, 'provider': self.provider.lower()}))
            return helper.dict_to_camel_case(cur.fetchone())

    @abstractmethod
    def get_obfuscated(self):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def update(self, changes, obfuscate=False):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def _add(self, data):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def delete(self):
        if False:
            return 10
        pass

    @abstractmethod
    def add_edit(self, data):
        if False:
            i = 10
            return i + 15
        pass