"""OS Login hooks.

.. spelling:word-list::
    ImportSshPublicKeyResponse
    oslogin
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from google.api_core.gapic_v1.method import DEFAULT, _MethodDefault
from google.cloud.oslogin_v1 import ImportSshPublicKeyResponse, OsLoginServiceClient
from airflow.providers.google.common.consts import CLIENT_INFO
from airflow.providers.google.common.hooks.base_google import PROVIDE_PROJECT_ID, GoogleBaseHook
if TYPE_CHECKING:
    from google.api_core.retry import Retry

class OSLoginHook(GoogleBaseHook):
    """
    Hook for Google OS login APIs.

    All the methods in the hook where project_id is used must be called with
    keyword arguments rather than positional.
    """

    def __init__(self, gcp_conn_id: str='google_cloud_default', impersonation_chain: str | Sequence[str] | None=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        if kwargs.get('delegate_to') is not None:
            raise RuntimeError('The `delegate_to` parameter has been deprecated before and finally removed in this version of Google Provider. You MUST convert it to `impersonate_chain`')
        super().__init__(gcp_conn_id=gcp_conn_id, impersonation_chain=impersonation_chain)
        self._conn: OsLoginServiceClient | None = None

    def get_conn(self) -> OsLoginServiceClient:
        if False:
            for i in range(10):
                print('nop')
        'Return OS Login service client.'
        if self._conn:
            return self._conn
        self._conn = OsLoginServiceClient(credentials=self.get_credentials(), client_info=CLIENT_INFO)
        return self._conn

    @GoogleBaseHook.fallback_to_default_project_id
    def import_ssh_public_key(self, user: str, ssh_public_key: dict, project_id: str=PROVIDE_PROJECT_ID, retry: Retry | _MethodDefault=DEFAULT, timeout: float | None=None, metadata: Sequence[tuple[str, str]]=()) -> ImportSshPublicKeyResponse:
        if False:
            print('Hello World!')
        '\n        Adds an SSH public key and returns the profile information.\n\n        Default POSIX account information is set when no username and UID exist as part of the login profile.\n\n        :param user: The unique ID for the user\n        :param ssh_public_key: The SSH public key and expiration time.\n        :param project_id: The project ID of the Google Cloud project.\n        :param retry: A retry object used to retry requests. If ``None`` is specified, requests will\n            be retried using a default configuration.\n        :param timeout: The amount of time, in seconds, to wait for the request to complete. Note that\n            if ``retry`` is specified, the timeout applies to each individual attempt.\n        :param metadata: Additional metadata that is provided to the method.\n        :return: A :class:`~google.cloud.oslogin_v1.ImportSshPublicKeyResponse` instance.\n        '
        conn = self.get_conn()
        return conn.import_ssh_public_key(request={'parent': f'users/{user}', 'ssh_public_key': ssh_public_key, 'project_id': project_id}, retry=retry, timeout=timeout, metadata=metadata)