"""This module contains a Google Cloud KMS hook."""
from __future__ import annotations
import base64
from typing import TYPE_CHECKING, Sequence
from google.api_core.gapic_v1.method import DEFAULT, _MethodDefault
from google.cloud.kms_v1 import KeyManagementServiceClient
from airflow.providers.google.common.consts import CLIENT_INFO
from airflow.providers.google.common.hooks.base_google import GoogleBaseHook
if TYPE_CHECKING:
    from google.api_core.retry import Retry

def _b64encode(s: bytes) -> str:
    if False:
        while True:
            i = 10
    'Base 64 encodes a bytes object to a string.'
    return base64.b64encode(s).decode('ascii')

def _b64decode(s: str) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    'Base 64 decodes a string to bytes.'
    return base64.b64decode(s.encode('utf-8'))

class CloudKMSHook(GoogleBaseHook):
    """
    Hook for Google Cloud Key Management service.

    :param gcp_conn_id: The connection ID to use when fetching connection info.
    :param impersonation_chain: Optional service account to impersonate using short-term
        credentials, or chained list of accounts required to get the access_token
        of the last account in the list, which will be impersonated in the request.
        If set as a string, the account must grant the originating account
        the Service Account Token Creator IAM role.
        If set as a sequence, the identities from the list must grant
        Service Account Token Creator IAM role to the directly preceding identity, with first
        account from the list granting this role to the originating account.
    """

    def __init__(self, gcp_conn_id: str='google_cloud_default', impersonation_chain: str | Sequence[str] | None=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        if kwargs.get('delegate_to') is not None:
            raise RuntimeError('The `delegate_to` parameter has been deprecated before and finally removed in this version of Google Provider. You MUST convert it to `impersonate_chain`')
        super().__init__(gcp_conn_id=gcp_conn_id, impersonation_chain=impersonation_chain)
        self._conn: KeyManagementServiceClient | None = None

    def get_conn(self) -> KeyManagementServiceClient:
        if False:
            return 10
        '\n        Retrieves connection to Cloud Key Management service.\n\n        :return: Cloud Key Management service object\n        '
        if not self._conn:
            self._conn = KeyManagementServiceClient(credentials=self.get_credentials(), client_info=CLIENT_INFO)
        return self._conn

    def encrypt(self, key_name: str, plaintext: bytes, authenticated_data: bytes | None=None, retry: Retry | _MethodDefault=DEFAULT, timeout: float | None=None, metadata: Sequence[tuple[str, str]]=()) -> str:
        if False:
            return 10
        '\n        Encrypts a plaintext message using Google Cloud KMS.\n\n        :param key_name: The Resource Name for the key (or key version)\n                         to be used for encryption. Of the form\n                         ``projects/*/locations/*/keyRings/*/cryptoKeys/**``\n        :param plaintext: The message to be encrypted.\n        :param authenticated_data: Optional additional authenticated data that\n                                   must also be provided to decrypt the message.\n        :param retry: A retry object used to retry requests. If None is specified, requests will not be\n            retried.\n        :param timeout: The amount of time, in seconds, to wait for the request to complete. Note that if\n            retry is specified, the timeout applies to each individual attempt.\n        :param metadata: Additional metadata that is provided to the method.\n        :return: The base 64 encoded ciphertext of the original message.\n        '
        response = self.get_conn().encrypt(request={'name': key_name, 'plaintext': plaintext, 'additional_authenticated_data': authenticated_data}, retry=retry, timeout=timeout, metadata=metadata)
        ciphertext = _b64encode(response.ciphertext)
        return ciphertext

    def decrypt(self, key_name: str, ciphertext: str, authenticated_data: bytes | None=None, retry: Retry | _MethodDefault=DEFAULT, timeout: float | None=None, metadata: Sequence[tuple[str, str]]=()) -> bytes:
        if False:
            while True:
                i = 10
        '\n        Decrypts a ciphertext message using Google Cloud KMS.\n\n        :param key_name: The Resource Name for the key to be used for decryption.\n                         Of the form ``projects/*/locations/*/keyRings/*/cryptoKeys/**``\n        :param ciphertext: The message to be decrypted.\n        :param authenticated_data: Any additional authenticated data that was\n                                   provided when encrypting the message.\n        :param retry: A retry object used to retry requests. If None is specified, requests will not be\n            retried.\n        :param timeout: The amount of time, in seconds, to wait for the request to complete. Note that if\n            retry is specified, the timeout applies to each individual attempt.\n        :param metadata: Additional metadata that is provided to the method.\n        :return: The original message.\n        '
        response = self.get_conn().decrypt(request={'name': key_name, 'ciphertext': _b64decode(ciphertext), 'additional_authenticated_data': authenticated_data}, retry=retry, timeout=timeout, metadata=metadata)
        return response.plaintext