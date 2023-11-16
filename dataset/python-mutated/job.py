from __future__ import annotations
import logging
import time
from typing import Any
import requests
from airflow.exceptions import AirflowException
from airflow.models import BaseOperator
from airflow.providers.plexus.hooks.plexus import PlexusHook
logger = logging.getLogger(__name__)

class PlexusJobOperator(BaseOperator):
    """
    Submits a Plexus job.

    :param job_params: parameters required to launch a job.

    Required job parameters are the following
        - "name": job name created by user.
        - "app": name of the application to run. found in Plexus UI.
        - "queue": public cluster name. found in Plexus UI.
        - "num_nodes": number of nodes.
        - "num_cores":  number of cores per node.

    """

    def __init__(self, job_params: dict, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.job_params = job_params
        self.required_params = {'name', 'app', 'queue', 'num_cores', 'num_nodes'}
        self.lookups = {'app': ('apps/', 'id', 'name'), 'billing_account_id': ('users/{}/billingaccounts/', 'id', None), 'queue': ('queues/', 'id', 'public_name')}
        self.job_params.update({'billing_account_id': None})
        self.is_service = None

    def execute(self, context: Any) -> Any:
        if False:
            print('Hello World!')
        hook = PlexusHook()
        params = self.construct_job_params(hook)
        if self.is_service is True:
            if self.job_params.get('expected_runtime') is None:
                end_state = 'Running'
            else:
                end_state = 'Finished'
        elif self.is_service is False:
            end_state = 'Completed'
        else:
            raise AirflowException('Unable to determine if application is running as a batch job or service. Contact Core Scientific AI Team.')
        logger.info('creating job w/ following params: %s', params)
        jobs_endpoint = hook.host + 'jobs/'
        headers = {'Authorization': f'Bearer {hook.token}'}
        create_job = requests.post(jobs_endpoint, headers=headers, data=params, timeout=5)
        if create_job.ok:
            job = create_job.json()
            jid = job['id']
            state = job['last_state']
            while state != end_state:
                time.sleep(3)
                jid_endpoint = jobs_endpoint + f'{jid}/'
                get_job = requests.get(jid_endpoint, headers=headers, timeout=5)
                if not get_job.ok:
                    raise AirflowException(f'Could not retrieve job status. Status Code: [{get_job.status_code}]. Reason: {get_job.reason} - {get_job.text}')
                new_state = get_job.json()['last_state']
                if new_state in ('Cancelled', 'Failed'):
                    raise AirflowException(f'Job {new_state}')
                elif new_state != state:
                    logger.info('job is %s', new_state)
                state = new_state
        else:
            raise AirflowException(f'Could not start job. Status Code: [{create_job.status_code}]. Reason: {create_job.reason} - {create_job.text}')

    def _api_lookup(self, param: str, hook):
        if False:
            return 10
        lookup = self.lookups[param]
        key = lookup[1]
        mapping = None if lookup[2] is None else (lookup[2], self.job_params[param])
        if param == 'billing_account_id':
            endpoint = hook.host + lookup[0].format(hook.user_id)
        else:
            endpoint = hook.host + lookup[0]
        headers = {'Authorization': f'Bearer {hook.token}'}
        response = requests.get(endpoint, headers=headers, timeout=5)
        results = response.json()['results']
        v = None
        if mapping is None:
            v = results[0][key]
        else:
            for dct in results:
                if dct[mapping[0]] == mapping[1]:
                    v = dct[key]
                if param == 'app':
                    self.is_service = dct['is_service']
        if v is None:
            raise AirflowException(f'Could not locate value for param:{key} at endpoint: {endpoint}')
        return v

    def construct_job_params(self, hook: Any) -> dict[Any, Any | None]:
        if False:
            print('Hello World!')
        '\n        Creates job_params dict for api call to launch a Plexus job.\n\n        Some parameters required to launch a job\n        are not available to the user in the Plexus\n        UI. For example, an app id is required, but\n        only the app name is provided in the UI.\n        This function acts as a backend lookup\n        of the required param value using the\n        user-provided value.\n\n        :param hook: plexus hook object\n        '
        missing_params = self.required_params - set(self.job_params)
        if missing_params:
            raise AirflowException(f"Missing the following required job_params: {', '.join(missing_params)}")
        params = {}
        for prm in self.job_params:
            if prm in self.lookups:
                v = self._api_lookup(param=prm, hook=hook)
                params[prm] = v
            else:
                params[prm] = self.job_params[prm]
        return params