from __future__ import annotations
import asyncio
from typing import Any, AsyncIterator, Sequence
from google.cloud.devtools.cloudbuild_v1.types import Build
from airflow.providers.google.cloud.hooks.cloud_build import CloudBuildAsyncHook
from airflow.triggers.base import BaseTrigger, TriggerEvent

class CloudBuildCreateBuildTrigger(BaseTrigger):
    """
    CloudBuildCreateBuildTrigger run on the trigger worker to perform create Build operation.

    :param id_: The ID of the build.
    :param project_id: Google Cloud Project where the job is running
    :param gcp_conn_id: Optional, the connection ID used to connect to Google Cloud Platform.
    :param impersonation_chain: Optional service account to impersonate using short-term
        credentials, or chained list of accounts required to get the access_token
        of the last account in the list, which will be impersonated in the request.
        If set as a string, the account must grant the originating account
        the Service Account Token Creator IAM role.
        If set as a sequence, the identities from the list must grant
        Service Account Token Creator IAM role to the directly preceding identity, with first
        account from the list granting this role to the originating account (templated).
    :param poll_interval: polling period in seconds to check for the status
    :param location: The location of the project.
    """

    def __init__(self, id_: str, project_id: str | None, gcp_conn_id: str='google_cloud_default', impersonation_chain: str | Sequence[str] | None=None, poll_interval: float=4.0, location: str='global'):
        if False:
            while True:
                i = 10
        super().__init__()
        self.id_ = id_
        self.project_id = project_id
        self.gcp_conn_id = gcp_conn_id
        self.impersonation_chain = impersonation_chain
        self.poll_interval = poll_interval
        self.location = location

    def serialize(self) -> tuple[str, dict[str, Any]]:
        if False:
            while True:
                i = 10
        'Serializes CloudBuildCreateBuildTrigger arguments and classpath.'
        return ('airflow.providers.google.cloud.triggers.cloud_build.CloudBuildCreateBuildTrigger', {'id_': self.id_, 'project_id': self.project_id, 'gcp_conn_id': self.gcp_conn_id, 'impersonation_chain': self.impersonation_chain, 'poll_interval': self.poll_interval, 'location': self.location})

    async def run(self) -> AsyncIterator[TriggerEvent]:
        """Gets current build execution status and yields a TriggerEvent."""
        hook = self._get_async_hook()
        try:
            while True:
                cloud_build_instance = await hook.get_cloud_build(id_=self.id_, project_id=self.project_id, location=self.location)
                if cloud_build_instance._pb.status in (Build.Status.SUCCESS,):
                    yield TriggerEvent({'instance': Build.to_dict(cloud_build_instance), 'id_': self.id_, 'status': 'success', 'message': 'Build completed'})
                    return
                elif cloud_build_instance._pb.status in (Build.Status.WORKING, Build.Status.PENDING, Build.Status.QUEUED):
                    self.log.info('Build is still running...')
                    self.log.info('Sleeping for %s seconds.', self.poll_interval)
                    await asyncio.sleep(self.poll_interval)
                elif cloud_build_instance._pb.status in (Build.Status.FAILURE, Build.Status.INTERNAL_ERROR, Build.Status.TIMEOUT, Build.Status.CANCELLED, Build.Status.EXPIRED):
                    yield TriggerEvent({'status': 'error', 'message': cloud_build_instance.status_detail})
                    return
                else:
                    yield TriggerEvent({'status': 'error', 'message': 'Unidentified status of Cloud Build instance'})
                    return
        except Exception as e:
            self.log.exception('Exception occurred while checking for Cloud Build completion')
            yield TriggerEvent({'status': 'error', 'message': str(e)})

    def _get_async_hook(self) -> CloudBuildAsyncHook:
        if False:
            return 10
        return CloudBuildAsyncHook(gcp_conn_id=self.gcp_conn_id)