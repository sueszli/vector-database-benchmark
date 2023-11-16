"""File containing abstract class for CPIX clients."""
import abc
import os
from typing import Dict, List
from google.cloud import secretmanager

class CpixClient(abc.ABC):
    """Abstract class for CPIX clients."""

    @abc.abstractmethod
    def fetch_keys(self, media_id: str, key_ids: List[str]) -> Dict[str, object]:
        if False:
            i = 10
            return i + 15
        'Fetches encryption keys and prepares JSON content to be written to Secret Manager.\n\n        Args:\n            media_id (string): Name for your asset, sometimes used by DRM providers to\n            show usage and reports.\n            key_ids (list[string]): List of IDs of any keys to fetch and prepare.\n\n        Returns:\n            Dictionary mapping key IDs to JSON-structured object containing key\n            information to be written to Secret Manager.\n        '

    @property
    @abc.abstractmethod
    def required_env_vars(self) -> List[str]:
        if False:
            print('Hello World!')
        'Returns environment variables which must be set to use the class.\n\n        The `PROJECT` env var is always required and does not need to be included\n        in the returned list.\n\n        Returns:\n            List of strings, names of environment variables which must be\n            set.\n        '

    def access_secret_version(self, secret_id: str, version_id: str) -> secretmanager.AccessSecretVersionResponse:
        if False:
            i = 10
            return i + 15
        'Fetches the content of the secret given secret id and version number\n\n        Args:\n            secret_id (string): Name of the secret.\n            version_id (string): Version number of the secret.\n\n        Returns:\n            secretmanager.AccessSecretVersionResponse object containing value of the\n            secret.\n        '
        client = secretmanager.SecretManagerServiceClient()
        project_id = os.environ.get('PROJECT')
        secret_name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
        response = client.access_secret_version(name=secret_name)
        return response.payload.data.decode().replace('\r\n', '\n')