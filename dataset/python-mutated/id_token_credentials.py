"""
You can execute this module to get ID Token.

    python -m airflow.providers.google.common.utils.id_token_credentials_provider

To obtain info about this token, run the following commands:

    ID_TOKEN="$(python -m airflow.providers.google.common.utils.id_token_credentials)"
    curl "https://www.googleapis.com/oauth2/v3/tokeninfo?id_token=${ID_TOKEN}" -v

.. spelling:word-list::

    RefreshError
"""
from __future__ import annotations
import json
import os
from typing import TYPE_CHECKING
import google.auth.transport
from google.auth import credentials as google_auth_credentials, environment_vars, exceptions
from google.oauth2 import credentials as oauth2_credentials, service_account
if TYPE_CHECKING:
    import google.oauth2
_AUTHORIZED_USER_TYPE = 'authorized_user'
_SERVICE_ACCOUNT_TYPE = 'service_account'
_EXTERNAL_ACCOUNT_TYPE = 'external_account'
_EXTERNAL_ACCOUNT_AUTHORIZED_USER_TYPE = 'external_account_authorized_user'
_IMPERSONATED_SERVICE_ACCOUNT_TYPE = 'impersonated_service_account'
_GDCH_SERVICE_ACCOUNT_TYPE = 'gdch_service_account'
_VALID_TYPES = (_AUTHORIZED_USER_TYPE, _SERVICE_ACCOUNT_TYPE, _EXTERNAL_ACCOUNT_TYPE, _EXTERNAL_ACCOUNT_AUTHORIZED_USER_TYPE, _IMPERSONATED_SERVICE_ACCOUNT_TYPE, _GDCH_SERVICE_ACCOUNT_TYPE)

class IDTokenCredentialsAdapter(google_auth_credentials.Credentials):
    """Convert Credentials with ``openid`` scope to IDTokenCredentials."""

    def __init__(self, credentials: oauth2_credentials.Credentials):
        if False:
            return 10
        super().__init__()
        self.credentials = credentials
        self.token = credentials.id_token

    @property
    def expired(self):
        if False:
            while True:
                i = 10
        return self.credentials.expired

    def refresh(self, request):
        if False:
            while True:
                i = 10
        self.credentials.refresh(request)
        self.token = self.credentials.id_token

def _load_credentials_from_file(filename: str, target_audience: str | None) -> google_auth_credentials.Credentials | None:
    if False:
        while True:
            i = 10
    '\n    Loads credentials from a file.\n\n    The credentials file must be a service account key or a stored authorized user credential.\n\n    :param filename: The full path to the credentials file.\n    :return: Loaded credentials\n    :raise google.auth.exceptions.DefaultCredentialsError: if the file is in the wrong format or is missing.\n    '
    if not os.path.exists(filename):
        raise exceptions.DefaultCredentialsError(f'File {filename} was not found.')
    with open(filename) as file_obj:
        try:
            info = json.load(file_obj)
        except json.JSONDecodeError:
            raise exceptions.DefaultCredentialsError(f'File {filename} is not a valid json file.')
    credential_type = info.get('type')
    if credential_type == _AUTHORIZED_USER_TYPE:
        current_credentials = oauth2_credentials.Credentials.from_authorized_user_info(info, scopes=['openid', 'email'])
        current_credentials = IDTokenCredentialsAdapter(credentials=current_credentials)
        return current_credentials
    elif credential_type == _SERVICE_ACCOUNT_TYPE:
        try:
            return service_account.IDTokenCredentials.from_service_account_info(info, target_audience=target_audience)
        except ValueError:
            raise exceptions.DefaultCredentialsError(f'Failed to load service account credentials from {filename}')
    raise exceptions.DefaultCredentialsError(f'The file {filename} does not have a valid type. Type is {credential_type}, expected one of {_VALID_TYPES}.')

def _get_explicit_environ_credentials(target_audience: str | None) -> google_auth_credentials.Credentials | None:
    if False:
        while True:
            i = 10
    'Gets credentials from the GOOGLE_APPLICATION_CREDENTIALS environment variable.'
    explicit_file = os.environ.get(environment_vars.CREDENTIALS)
    if explicit_file is None:
        return None
    current_credentials = _load_credentials_from_file(os.environ[environment_vars.CREDENTIALS], target_audience=target_audience)
    return current_credentials

def _get_gcloud_sdk_credentials(target_audience: str | None) -> google_auth_credentials.Credentials | None:
    if False:
        i = 10
        return i + 15
    'Gets the credentials and project ID from the Cloud SDK.'
    from google.auth import _cloud_sdk
    credentials_filename = _cloud_sdk.get_application_default_credentials_path()
    if not os.path.isfile(credentials_filename):
        return None
    current_credentials = _load_credentials_from_file(credentials_filename, target_audience)
    return current_credentials

def _get_gce_credentials(target_audience: str | None, request: google.auth.transport.Request | None=None) -> google_auth_credentials.Credentials | None:
    if False:
        i = 10
        return i + 15
    'Gets credentials and project ID from the GCE Metadata Service.'
    try:
        from google.auth import compute_engine
        from google.auth.compute_engine import _metadata
    except ImportError:
        return None
    from google.auth.transport import _http_client
    if request is None:
        request = _http_client.Request()
    if _metadata.ping(request=request):
        return compute_engine.IDTokenCredentials(request, target_audience, use_metadata_identity_endpoint=True)
    return None

def get_default_id_token_credentials(target_audience: str | None, request: google.auth.transport.Request=None) -> google_auth_credentials.Credentials:
    if False:
        return 10
    "Gets the default ID Token credentials for the current environment.\n\n    `Application Default Credentials`_ provides an easy way to obtain credentials to call Google APIs for\n    server-to-server or local applications.\n\n    .. _Application Default Credentials: https://developers.google.com        /identity/protocols/application-default-credentials\n\n    :param target_audience: The intended audience for these credentials.\n    :param request: An object used to make HTTP requests. This is used to detect whether the application\n            is running on Compute Engine. If not specified, then it will use the standard library http client\n            to make requests.\n    :return: the current environment's credentials.\n    :raises ~google.auth.exceptions.DefaultCredentialsError:\n        If no credentials were found, or if the credentials found were invalid.\n    "
    checkers = (lambda : _get_explicit_environ_credentials(target_audience), lambda : _get_gcloud_sdk_credentials(target_audience), lambda : _get_gce_credentials(target_audience, request))
    for checker in checkers:
        current_credentials = checker()
        if current_credentials is not None:
            return current_credentials
    raise exceptions.DefaultCredentialsError(f'Could not automatically determine credentials. Please set {environment_vars.CREDENTIALS} or\n        explicitly create credentials and re-run the application. For more information, please see\n        https://cloud.google.com/docs/authentication/getting-started\n'.strip())
if __name__ == '__main__':
    from google.auth.transport import requests
    request_adapter = requests.Request()
    creds = get_default_id_token_credentials(target_audience=None)
    creds.refresh(request=request_adapter)
    print(creds.token)