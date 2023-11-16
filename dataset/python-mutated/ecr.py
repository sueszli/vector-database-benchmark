from __future__ import annotations
import base64
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseHook
from airflow.utils.log.secrets_masker import mask_secret
if TYPE_CHECKING:
    from datetime import datetime
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EcrCredentials:
    """Helper (frozen dataclass) for storing temporary ECR credentials."""
    username: str
    password: str
    proxy_endpoint: str
    expires_at: datetime

    def __post_init__(self):
        if False:
            print('Hello World!')
        mask_secret(self.password)
        logger.debug('Credentials to Amazon ECR %r expires at %s.', self.proxy_endpoint, self.expires_at)

    @property
    def registry(self) -> str:
        if False:
            print('Hello World!')
        'Return registry in appropriate `docker login` format.'
        return self.proxy_endpoint.replace('https://', '')

class EcrHook(AwsBaseHook):
    """
    Interact with Amazon Elastic Container Registry (ECR).

    Provide thin wrapper around :external+boto3:py:class:`boto3.client("ecr") <ECR.Client>`.

    Additional arguments (such as ``aws_conn_id``) may be specified and
    are passed down to the underlying AwsBaseHook.

    .. seealso::
        - :class:`airflow.providers.amazon.aws.hooks.base_aws.AwsBaseHook`
    """

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        kwargs['client_type'] = 'ecr'
        super().__init__(**kwargs)

    def get_temporary_credentials(self, registry_ids: list[str] | str | None=None) -> list[EcrCredentials]:
        if False:
            return 10
        'Get temporary credentials for Amazon ECR.\n\n        .. seealso::\n            - :external+boto3:py:meth:`ECR.Client.get_authorization_token`\n\n        :param registry_ids: Either AWS Account ID or list of AWS Account IDs that are associated\n            with the registries from which credentials are obtained. If you do not specify a registry,\n            the default registry is assumed.\n        :return: list of :class:`airflow.providers.amazon.aws.hooks.ecr.EcrCredentials`,\n            obtained credentials valid for 12 hours.\n        '
        registry_ids = registry_ids or None
        if isinstance(registry_ids, str):
            registry_ids = [registry_ids]
        if registry_ids:
            response = self.conn.get_authorization_token(registryIds=registry_ids)
        else:
            response = self.conn.get_authorization_token()
        creds = []
        for auth_data in response['authorizationData']:
            (username, password) = base64.b64decode(auth_data['authorizationToken']).decode('utf-8').split(':')
            creds.append(EcrCredentials(username=username, password=password, proxy_endpoint=auth_data['proxyEndpoint'], expires_at=auth_data['expiresAt']))
        return creds