from __future__ import annotations
from typing import Any, Sequence
from googleapiclient.discovery import Resource, build
from airflow.exceptions import AirflowException
from airflow.providers.google.common.hooks.base_google import GoogleBaseHook

class GoogleDeploymentManagerHook(GoogleBaseHook):
    """
    Interact with Google Cloud Deployment Manager using the Google Cloud connection.

    This allows for scheduled and programmatic inspection and deletion of resources managed by GDM.
    """

    def __init__(self, gcp_conn_id: str='google_cloud_default', impersonation_chain: str | Sequence[str] | None=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        if kwargs.get('delegate_to') is not None:
            raise RuntimeError('The `delegate_to` parameter has been deprecated before and finally removed in this version of Google Provider. You MUST convert it to `impersonate_chain`')
        super().__init__(gcp_conn_id=gcp_conn_id, impersonation_chain=impersonation_chain)

    def get_conn(self) -> Resource:
        if False:
            for i in range(10):
                print('nop')
        'Returns a Google Deployment Manager service object.'
        http_authorized = self._authorize()
        return build('deploymentmanager', 'v2', http=http_authorized, cache_discovery=False)

    @GoogleBaseHook.fallback_to_default_project_id
    def list_deployments(self, project_id: str | None=None, deployment_filter: str | None=None, order_by: str | None=None) -> list[dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Lists deployments in a google cloud project.\n\n        :param project_id: The project ID for this request.\n        :param deployment_filter: A filter expression which limits resources returned in the response.\n        :param order_by: A field name to order by, ex: "creationTimestamp desc"\n        '
        deployments: list[dict] = []
        conn = self.get_conn()
        request = conn.deployments().list(project=project_id, filter=deployment_filter, orderBy=order_by)
        while request is not None:
            response = request.execute(num_retries=self.num_retries)
            deployments.extend(response.get('deployments', []))
            request = conn.deployments().list_next(previous_request=request, previous_response=response)
        return deployments

    @GoogleBaseHook.fallback_to_default_project_id
    def delete_deployment(self, project_id: str | None, deployment: str | None=None, delete_policy: str | None=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Deletes a deployment and all associated resources in a google cloud project.\n\n        :param project_id: The project ID for this request.\n        :param deployment: The name of the deployment for this request.\n        :param delete_policy: Sets the policy to use for deleting resources. (ABANDON | DELETE)\n        '
        conn = self.get_conn()
        request = conn.deployments().delete(project=project_id, deployment=deployment, deletePolicy=delete_policy)
        resp = request.execute()
        if 'error' in resp.keys():
            raise AirflowException('Errors deleting deployment: ', ', '.join((err['message'] for err in resp['error']['errors'])))