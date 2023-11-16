from __future__ import annotations
import re
from functools import cached_property
from typing import TYPE_CHECKING
from google.api_core.exceptions import InvalidArgument, NotFound, PermissionDenied
from google.cloud.secretmanager_v1 import SecretManagerServiceClient
from airflow.providers.google.common.consts import CLIENT_INFO
from airflow.utils.log.logging_mixin import LoggingMixin
if TYPE_CHECKING:
    import google
SECRET_ID_PATTERN = '^[a-zA-Z0-9-_]*$'

class _SecretManagerClient(LoggingMixin):
    """Retrieve Secrets object from Google Cloud Secrets Manager.

    This is a common class reused between SecretsManager and Secrets Hook that
    provides the shared authentication and verification mechanisms. This class
    should not be used directly; use SecretsManager or SecretsHook instead.

    :param credentials: Credentials used to authenticate to GCP
    """

    def __init__(self, credentials: google.auth.credentials.Credentials) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.credentials = credentials

    @staticmethod
    def is_valid_secret_name(secret_name: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Whether the secret name is valid.\n\n        :param secret_name: name of the secret\n        '
        return bool(re.match(SECRET_ID_PATTERN, secret_name))

    @cached_property
    def client(self) -> SecretManagerServiceClient:
        if False:
            while True:
                i = 10
        'Create an authenticated KMS client.'
        _client = SecretManagerServiceClient(credentials=self.credentials, client_info=CLIENT_INFO)
        return _client

    def get_secret(self, secret_id: str, project_id: str, secret_version: str='latest') -> str | None:
        if False:
            print('Hello World!')
        "Get secret value from the Secret Manager.\n\n        :param secret_id: Secret Key\n        :param project_id: Project id to use\n        :param secret_version: version of the secret (default is 'latest')\n        "
        name = self.client.secret_version_path(project_id, secret_id, secret_version)
        try:
            response = self.client.access_secret_version(request={'name': name})
            value = response.payload.data.decode('UTF-8')
            return value
        except NotFound:
            self.log.debug('Google Cloud API Call Error (NotFound): Secret ID %s not found.', secret_id)
            return None
        except PermissionDenied:
            self.log.error("Google Cloud API Call Error (PermissionDenied): No access for Secret ID %s.\n                Did you add 'secretmanager.versions.access' permission?", secret_id)
            return None
        except InvalidArgument:
            self.log.error('Google Cloud API Call Error (InvalidArgument): Invalid secret ID %s.\n                Only ASCII alphabets (a-Z), numbers (0-9), dashes (-), and underscores (_)\n                are allowed in the secret ID.\n                ', secret_id)
            return None