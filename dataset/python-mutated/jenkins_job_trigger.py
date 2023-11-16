from __future__ import annotations
import ast
import json
import socket
import time
from functools import cached_property
from typing import Any, Iterable, Mapping, Sequence, Union
from urllib.error import HTTPError, URLError
import jenkins
from deprecated.classic import deprecated
from jenkins import Jenkins, JenkinsException
from requests import Request
from airflow.exceptions import AirflowException, AirflowProviderDeprecationWarning
from airflow.models import BaseOperator
from airflow.providers.jenkins.hooks.jenkins import JenkinsHook
JenkinsRequest = Mapping[str, Any]
ParamType = Union[str, dict, list, None]

def jenkins_request_with_headers(jenkins_server: Jenkins, req: Request) -> JenkinsRequest | None:
    if False:
        i = 10
        return i + 15
    'Create a Jenkins request from a raw request.\n\n    We need to get the headers in addition to the body answer to get the\n    location from them. This function uses ``jenkins_request`` from\n    python-jenkins with just the return call changed.\n\n    :param jenkins_server: The server to query\n    :param req: The request to execute\n    :return: Dict containing the response body (key body)\n        and the headers coming along (headers)\n    '
    try:
        response = jenkins_server.jenkins_request(req)
        response_body = response.content
        response_headers = response.headers
        if response_body is None:
            raise jenkins.EmptyResponseException(f'Error communicating with server[{jenkins_server.server}]: empty response')
        return {'body': response_body.decode('utf-8'), 'headers': response_headers}
    except HTTPError as e:
        if e.code in [401, 403, 500]:
            raise JenkinsException(f'Error in request. Possibly authentication failed [{e.code}]: {e.reason}')
        elif e.code == 404:
            raise jenkins.NotFoundException('Requested item could not be found')
        else:
            raise
    except socket.timeout as e:
        raise jenkins.TimeoutException(f'Error in request: {e}')
    except URLError as e:
        raise JenkinsException(f'Error in request: {e.reason}')
    return None

class JenkinsJobTriggerOperator(BaseOperator):
    """Trigger a Jenkins Job and monitor its execution.

    This operator depend on the python-jenkins library version >= 0.4.15 to
    communicate with the Jenkins server. You'll also need to configure a Jenkins
    connection in the connections screen.

    :param jenkins_connection_id: The jenkins connection to use for this job
    :param job_name: The name of the job to trigger
    :param parameters: The parameters block provided to jenkins for use in
        the API call when triggering a build. (templated)
    :param sleep_time: How long will the operator sleep between each status
        request for the job (min 1, default 10)
    :param max_try_before_job_appears: The maximum number of requests to make
        while waiting for the job to appears on jenkins server (default 10)
    :param allowed_jenkins_states: Iterable of allowed result jenkins states, default is ``['SUCCESS']``
    """
    template_fields: Sequence[str] = ('parameters',)
    template_ext: Sequence[str] = ('.json',)
    ui_color = '#f9ec86'

    def __init__(self, *, jenkins_connection_id: str, job_name: str, parameters: ParamType=None, sleep_time: int=10, max_try_before_job_appears: int=10, allowed_jenkins_states: Iterable[str] | None=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.job_name = job_name
        self.parameters = parameters
        self.sleep_time = max(sleep_time, 1)
        self.jenkins_connection_id = jenkins_connection_id
        self.max_try_before_job_appears = max_try_before_job_appears
        self.allowed_jenkins_states = list(allowed_jenkins_states) if allowed_jenkins_states else ['SUCCESS']

    def build_job(self, jenkins_server: Jenkins, params: ParamType=None) -> JenkinsRequest | None:
        if False:
            for i in range(10):
                print('nop')
        'Trigger a build job.\n\n        This returns a dict with 2 keys ``body`` and ``headers``. ``headers``\n        contains also a dict-like object which can be queried to get the\n        location to poll in the queue.\n\n        :param jenkins_server: The jenkins server where the job should be triggered\n        :param params: The parameters block to provide to jenkins API call.\n        :return: Dict containing the response body (key body)\n            and the headers coming along (headers)\n        '
        if params and isinstance(params, str):
            params = ast.literal_eval(params)
        request = Request(method='POST', url=jenkins_server.build_job_url(self.job_name, params, None))
        return jenkins_request_with_headers(jenkins_server, request)

    def poll_job_in_queue(self, location: str, jenkins_server: Jenkins) -> int:
        if False:
            while True:
                i = 10
        'Poll the jenkins queue until the job is executed.\n\n        When we trigger a job through an API call, the job is first put in the\n        queue without having a build number assigned. We have to wait until the\n        job exits the queue to know its build number.\n\n        To do so, we add ``/api/json`` (or ``/api/xml``) to the location\n        returned by the ``build_job`` call, and poll this file. When an\n        ``executable`` block appears in the response, the job execution would\n        have started, and the field ``number`` would contains the build number.\n\n        :param location: Location to poll, returned in the header of the build_job call\n        :param jenkins_server: The jenkins server to poll\n        :return: The build_number corresponding to the triggered job\n        '
        location += '/api/json'
        self.log.info('Polling jenkins queue at the url %s', location)
        for attempt in range(self.max_try_before_job_appears):
            if attempt:
                time.sleep(self.sleep_time)
            try:
                location_answer = jenkins_request_with_headers(jenkins_server, Request(method='POST', url=location))
            except (HTTPError, JenkinsException):
                self.log.warning('polling failed, retrying', exc_info=True)
            else:
                if location_answer is not None:
                    json_response = json.loads(location_answer['body'])
                    if 'executable' in json_response and json_response['executable'] is not None and ('number' in json_response['executable']):
                        build_number = json_response['executable']['number']
                        self.log.info('Job executed on Jenkins side with the build number %s', build_number)
                        return build_number
        else:
            raise AirflowException(f"The job hasn't been executed after polling the queue {self.max_try_before_job_appears} times")

    @cached_property
    def hook(self) -> JenkinsHook:
        if False:
            i = 10
            return i + 15
        'Instantiate the Jenkins hook.'
        return JenkinsHook(self.jenkins_connection_id)

    @deprecated(reason='use `hook` property instead.', category=AirflowProviderDeprecationWarning)
    def get_hook(self) -> JenkinsHook:
        if False:
            for i in range(10):
                print('nop')
        'Instantiate the Jenkins hook.'
        return self.hook

    def execute(self, context: Mapping[Any, Any]) -> str | None:
        if False:
            return 10
        self.log.info('Triggering the job %s on the jenkins : %s with the parameters : %s', self.job_name, self.jenkins_connection_id, self.parameters)
        jenkins_server = self.hook.get_jenkins_server()
        jenkins_response = self.build_job(jenkins_server, self.parameters)
        if jenkins_response:
            build_number = self.poll_job_in_queue(jenkins_response['headers']['Location'], jenkins_server)
        time.sleep(self.sleep_time)
        keep_polling_job = True
        build_info = None
        try:
            while keep_polling_job:
                build_info = jenkins_server.get_build_info(name=self.job_name, number=build_number)
                if build_info['result'] is not None:
                    keep_polling_job = False
                    if build_info['result'] not in self.allowed_jenkins_states:
                        raise AirflowException(f"Jenkins job failed, final state : {build_info['result']}. Find more information on job url : {build_info['url']}")
                else:
                    self.log.info('Waiting for job to complete : %s , build %s', self.job_name, build_number)
                    time.sleep(self.sleep_time)
        except jenkins.NotFoundException as err:
            raise AirflowException(f'Jenkins job status check failed. Final error was: {err.resp.status}')
        except jenkins.JenkinsException as err:
            raise AirflowException(f'Jenkins call failed with error : {err}, if you have parameters double check them, jenkins sends back this exception for unknown parametersYou can also check logs for more details on this exception (jenkins_url/log/rss)')
        if build_info:
            return build_info['url']
        return None