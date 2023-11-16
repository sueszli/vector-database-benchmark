from __future__ import annotations
from typing import TYPE_CHECKING, Iterable
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.providers.jenkins.hooks.jenkins import JenkinsHook
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class JenkinsBuildSensor(BaseSensorOperator):
    """Monitor a Jenkins job and pass when it is finished building.

    This is regardless of the build outcome.

    :param jenkins_connection_id: The jenkins connection to use for this job
    :param job_name: The name of the job to check
    :param build_number: Build number to check - if None, the latest build will be used
    """

    def __init__(self, *, jenkins_connection_id: str, job_name: str, build_number: int | None=None, target_states: Iterable[str] | None=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.job_name = job_name
        self.build_number = build_number
        self.jenkins_connection_id = jenkins_connection_id
        self.target_states = target_states or ['SUCCESS', 'FAILED']

    def poke(self, context: Context) -> bool:
        if False:
            while True:
                i = 10
        self.log.info('Poking jenkins job %s', self.job_name)
        hook = JenkinsHook(self.jenkins_connection_id)
        build_number = self.build_number or hook.get_latest_build_number(self.job_name)
        is_building = hook.get_build_building_state(self.job_name, build_number)
        if is_building:
            self.log.info('Build still ongoing!')
            return False
        build_result = hook.get_build_result(self.job_name, build_number)
        self.log.info('Build is finished, result is %s', 'build_result')
        if build_result in self.target_states:
            return True
        else:
            message = f'Build {build_number} finished with a result {build_result}, which does not meet the target state {self.target_states}.'
            if self.soft_fail:
                raise AirflowSkipException(message)
            raise AirflowException(message)