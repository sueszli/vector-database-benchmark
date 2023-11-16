"""Objects relating to sourcing connections from Google Cloud Secrets Manager."""
from __future__ import annotations
import logging
import re
import warnings
from google.auth.exceptions import DefaultCredentialsError
from airflow.exceptions import AirflowException, AirflowProviderDeprecationWarning
from airflow.providers.google.cloud._internal_client.secret_manager_client import _SecretManagerClient
from airflow.providers.google.cloud.utils.credentials_provider import get_credentials_and_project_id
from airflow.secrets import BaseSecretsBackend
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.version import version as airflow_version
log = logging.getLogger(__name__)
SECRET_ID_PATTERN = '^[a-zA-Z0-9-_]*$'

def _parse_version(val):
    if False:
        for i in range(10):
            print('nop')
    match = re.search('(\\d+)\\.(\\d+)\\.(\\d+)', val)
    return tuple((int(x) for x in match.groups()))

class CloudSecretManagerBackend(BaseSecretsBackend, LoggingMixin):
    """
    Retrieves Connection object from Google Cloud Secrets Manager.

    Configurable via ``airflow.cfg`` as follows:

    .. code-block:: ini

        [secrets]
        backend = airflow.providers.google.cloud.secrets.secret_manager.CloudSecretManagerBackend
        backend_kwargs = {"connections_prefix": "airflow-connections", "sep": "-"}

    For example, if the Secrets Manager secret id is ``airflow-connections-smtp_default``, this would be
    accessible if you provide ``{"connections_prefix": "airflow-connections", "sep": "-"}`` and request
    conn_id ``smtp_default``.

    If the Secrets Manager secret id is ``airflow-variables-hello``, this would be
    accessible if you provide ``{"variables_prefix": "airflow-variables", "sep": "-"}`` and request
    Variable Key ``hello``.

    The full secret id should follow the pattern "[a-zA-Z0-9-_]".

    :param connections_prefix: Specifies the prefix of the secret to read to get Connections.
        If set to None (null), requests for connections will not be sent to GCP Secrets Manager
    :param variables_prefix: Specifies the prefix of the secret to read to get Variables.
        If set to None (null), requests for variables will not be sent to GCP Secrets Manager
    :param config_prefix: Specifies the prefix of the secret to read to get Airflow Configurations
        containing secrets.
        If set to None (null), requests for configurations will not be sent to GCP Secrets Manager
    :param gcp_key_path: Path to Google Cloud Service Account key file (JSON). Mutually exclusive with
        gcp_keyfile_dict. use default credentials in the current environment if not provided.
    :param gcp_keyfile_dict: Dictionary of keyfile parameters. Mutually exclusive with gcp_key_path.
    :param gcp_credential_config_file: File path to or content of a GCP credential configuration file.
    :param gcp_scopes: Comma-separated string containing OAuth2 scopes
    :param project_id: Project ID to read the secrets from. If not passed, the project ID from credentials
        will be used.
    :param sep: Separator used to concatenate connections_prefix and conn_id. Default: "-"
    """

    def __init__(self, connections_prefix: str='airflow-connections', variables_prefix: str='airflow-variables', config_prefix: str='airflow-config', gcp_keyfile_dict: dict | None=None, gcp_key_path: str | None=None, gcp_credential_config_file: dict[str, str] | str | None=None, gcp_scopes: str | None=None, project_id: str | None=None, sep: str='-', **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.connections_prefix = connections_prefix
        self.variables_prefix = variables_prefix
        self.config_prefix = config_prefix
        self.sep = sep
        if connections_prefix is not None:
            if not self._is_valid_prefix_and_sep():
                raise AirflowException(f'`connections_prefix`, `variables_prefix` and `sep` should follows that pattern {SECRET_ID_PATTERN}')
        try:
            (self.credentials, self.project_id) = get_credentials_and_project_id(keyfile_dict=gcp_keyfile_dict, key_path=gcp_key_path, credential_config_file=gcp_credential_config_file, scopes=gcp_scopes)
        except (DefaultCredentialsError, FileNotFoundError):
            log.exception('Unable to load credentials for GCP Secret Manager. Make sure that the keyfile path or dictionary, credential configuration file, or GOOGLE_APPLICATION_CREDENTIALS environment variable is correct and properly configured.')
        if project_id:
            self.project_id = project_id

    @property
    def client(self) -> _SecretManagerClient:
        if False:
            while True:
                i = 10
        '\n        Property returning secret client.\n\n        :return: Secrets client\n        '
        return _SecretManagerClient(credentials=self.credentials)

    def _is_valid_prefix_and_sep(self) -> bool:
        if False:
            print('Hello World!')
        prefix = self.connections_prefix + self.sep
        return _SecretManagerClient.is_valid_secret_name(prefix)

    def get_conn_value(self, conn_id: str) -> str | None:
        if False:
            print('Hello World!')
        '\n        Get serialized representation of Connection.\n\n        :param conn_id: connection id\n        '
        if self.connections_prefix is None:
            return None
        return self._get_secret(self.connections_prefix, conn_id)

    def get_conn_uri(self, conn_id: str) -> str | None:
        if False:
            i = 10
            return i + 15
        '\n        Return URI representation of Connection conn_id.\n\n        As of Airflow version 2.3.0 this method is deprecated.\n\n        :param conn_id: the connection id\n        :return: deserialized Connection\n        '
        if _parse_version(airflow_version) >= (2, 3):
            warnings.warn(f'Method `{self.__class__.__name__}.get_conn_uri` is deprecated and will be removed in a future release.  Please use method `get_conn_value` instead.', AirflowProviderDeprecationWarning, stacklevel=2)
        return self.get_conn_value(conn_id)

    def get_variable(self, key: str) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get Airflow Variable from Environment Variable.\n\n        :param key: Variable Key\n        :return: Variable Value\n        '
        if self.variables_prefix is None:
            return None
        return self._get_secret(self.variables_prefix, key)

    def get_config(self, key: str) -> str | None:
        if False:
            return 10
        '\n        Get Airflow Configuration.\n\n        :param key: Configuration Option Key\n        :return: Configuration Option Value\n        '
        if self.config_prefix is None:
            return None
        return self._get_secret(self.config_prefix, key)

    def _get_secret(self, path_prefix: str, secret_id: str) -> str | None:
        if False:
            i = 10
            return i + 15
        '\n        Get secret value from the SecretManager based on prefix.\n\n        :param path_prefix: Prefix for the Path to get Secret\n        :param secret_id: Secret Key\n        '
        secret_id = self.build_path(path_prefix, secret_id, self.sep)
        return self.client.get_secret(secret_id=secret_id, project_id=self.project_id)