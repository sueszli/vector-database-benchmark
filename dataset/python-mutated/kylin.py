from __future__ import annotations
from kylinpy import exceptions, kylinpy
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook

class KylinHook(BaseHook):
    """
    Interact with Kylin to run CubeSource commands and get job status.

    :param kylin_conn_id: The connection id as configured in Airflow administration.
    :param project: project name
    :param dsn: dsn
    """

    def __init__(self, kylin_conn_id: str='kylin_default', project: str | None=None, dsn: str | None=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.kylin_conn_id = kylin_conn_id
        self.project = project
        self.dsn = dsn

    def get_conn(self):
        if False:
            i = 10
            return i + 15
        conn = self.get_connection(self.kylin_conn_id)
        if self.dsn:
            return kylinpy.create_kylin(self.dsn)
        self.project = self.project or conn.schema
        return kylinpy.Kylin(conn.host, username=conn.login, password=conn.password, port=conn.port, project=self.project, **conn.extra_dejson)

    def cube_run(self, datasource_name, op, **op_args):
        if False:
            print('Hello World!')
        '\n        Run CubeSource command which in CubeSource.support_invoke_command.\n\n        :param datasource_name:\n        :param op: command\n        :param op_args: command args\n        :return: response\n        '
        cube_source = self.get_conn().get_datasource(datasource_name)
        try:
            return cube_source.invoke_command(op, **op_args)
        except exceptions.KylinError as err:
            raise AirflowException(f'Cube operation {op} error , Message: {err}')

    def get_job_status(self, job_id):
        if False:
            return 10
        '\n        Get job status.\n\n        :param job_id: kylin job id\n        :return: job status\n        '
        return self.get_conn().get_job(job_id).status