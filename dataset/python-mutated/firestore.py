"""Hook for Google Cloud Firestore service."""
from __future__ import annotations
import time
from typing import Sequence
from googleapiclient.discovery import build, build_from_document
from airflow.exceptions import AirflowException
from airflow.providers.google.common.hooks.base_google import GoogleBaseHook
TIME_TO_SLEEP_IN_SECONDS = 5

class CloudFirestoreHook(GoogleBaseHook):
    """
    Hook for the Google Firestore APIs.

    All the methods in the hook where project_id is used must be called with
    keyword arguments rather than positional.

    :param api_version: API version used (for example v1 or v1beta1).
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
    _conn: build | None = None

    def __init__(self, api_version: str='v1', gcp_conn_id: str='google_cloud_default', impersonation_chain: str | Sequence[str] | None=None) -> None:
        if False:
            return 10
        super().__init__(gcp_conn_id=gcp_conn_id, impersonation_chain=impersonation_chain)
        self.api_version = api_version

    def get_conn(self):
        if False:
            i = 10
            return i + 15
        '\n        Retrieves the connection to Cloud Firestore.\n\n        :return: Google Cloud Firestore services object.\n        '
        if not self._conn:
            http_authorized = self._authorize()
            non_authorized_conn = build('firestore', self.api_version, cache_discovery=False)
            self._conn = build_from_document(non_authorized_conn._rootDesc, http=http_authorized)
        return self._conn

    @GoogleBaseHook.fallback_to_default_project_id
    def export_documents(self, body: dict, database_id: str='(default)', project_id: str | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Starts a export with the specified configuration.\n\n        :param database_id: The Database ID.\n        :param body: The request body.\n            See:\n            https://firebase.google.com/docs/firestore/reference/rest/v1beta1/projects.databases/exportDocuments\n        :param project_id: Optional, Google Cloud Project project_id where the database belongs.\n            If set to None or missing, the default project_id from the Google Cloud connection is used.\n        '
        service = self.get_conn()
        name = f'projects/{project_id}/databases/{database_id}'
        operation = service.projects().databases().exportDocuments(name=name, body=body).execute(num_retries=self.num_retries)
        self._wait_for_operation_to_complete(operation['name'])

    def _wait_for_operation_to_complete(self, operation_name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Waits for the named operation to complete - checks status of the asynchronous call.\n\n        :param operation_name: The name of the operation.\n        :return: The response returned by the operation.\n        :exception: AirflowException in case error is returned.\n        '
        service = self.get_conn()
        while True:
            operation_response = service.projects().databases().operations().get(name=operation_name).execute(num_retries=self.num_retries)
            if operation_response.get('done'):
                response = operation_response.get('response')
                error = operation_response.get('error')
                if error:
                    raise AirflowException(str(error))
                return response
            time.sleep(TIME_TO_SLEEP_IN_SECONDS)