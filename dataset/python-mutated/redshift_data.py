from __future__ import annotations
import time
from pprint import pformat
from typing import TYPE_CHECKING, Any, Iterable
from airflow.providers.amazon.aws.hooks.base_aws import AwsGenericHook
from airflow.providers.amazon.aws.utils import trim_none_values
if TYPE_CHECKING:
    from mypy_boto3_redshift_data import RedshiftDataAPIServiceClient

class RedshiftDataHook(AwsGenericHook['RedshiftDataAPIServiceClient']):
    """
    Interact with Amazon Redshift Data API.

    Provide thin wrapper around
    :external+boto3:py:class:`boto3.client("redshift-data") <RedshiftDataAPIService.Client>`.

    Additional arguments (such as ``aws_conn_id``) may be specified and
    are passed down to the underlying AwsBaseHook.

    .. seealso::
        - :class:`airflow.providers.amazon.aws.hooks.base_aws.AwsBaseHook`
        - `Amazon Redshift Data API         <https://docs.aws.amazon.com/redshift-data/latest/APIReference/Welcome.html>`__
    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        kwargs['client_type'] = 'redshift-data'
        super().__init__(*args, **kwargs)

    def execute_query(self, database: str, sql: str | list[str], cluster_identifier: str | None=None, db_user: str | None=None, parameters: Iterable | None=None, secret_arn: str | None=None, statement_name: str | None=None, with_event: bool=False, wait_for_completion: bool=True, poll_interval: int=10, workgroup_name: str | None=None) -> str:
        if False:
            print('Hello World!')
        "\n        Execute a statement against Amazon Redshift.\n\n        :param database: the name of the database\n        :param sql: the SQL statement or list of  SQL statement to run\n        :param cluster_identifier: unique identifier of a cluster\n        :param db_user: the database username\n        :param parameters: the parameters for the SQL statement\n        :param secret_arn: the name or ARN of the secret that enables db access\n        :param statement_name: the name of the SQL statement\n        :param with_event: indicates whether to send an event to EventBridge\n        :param wait_for_completion: indicates whether to wait for a result, if True wait, if False don't wait\n        :param poll_interval: how often in seconds to check the query status\n        :param workgroup_name: name of the Redshift Serverless workgroup. Mutually exclusive with\n            `cluster_identifier`. Specify this parameter to query Redshift Serverless. More info\n            https://docs.aws.amazon.com/redshift/latest/mgmt/working-with-serverless.html\n\n        :returns statement_id: str, the UUID of the statement\n        "
        kwargs: dict[str, Any] = {'ClusterIdentifier': cluster_identifier, 'Database': database, 'DbUser': db_user, 'Parameters': parameters, 'WithEvent': with_event, 'SecretArn': secret_arn, 'StatementName': statement_name, 'WorkgroupName': workgroup_name}
        if isinstance(sql, list):
            kwargs['Sqls'] = sql
            resp = self.conn.batch_execute_statement(**trim_none_values(kwargs))
        else:
            kwargs['Sql'] = sql
            resp = self.conn.execute_statement(**trim_none_values(kwargs))
        statement_id = resp['Id']
        if bool(cluster_identifier) is bool(workgroup_name):
            raise ValueError("Either 'cluster_identifier' or 'workgroup_name' must be specified.")
        if wait_for_completion:
            self.wait_for_results(statement_id, poll_interval=poll_interval)
        return statement_id

    def wait_for_results(self, statement_id, poll_interval):
        if False:
            print('Hello World!')
        while True:
            self.log.info('Polling statement %s', statement_id)
            resp = self.conn.describe_statement(Id=statement_id)
            status = resp['Status']
            if status == 'FINISHED':
                num_rows = resp.get('ResultRows')
                if num_rows is not None:
                    self.log.info('Processed %s rows', num_rows)
                return status
            elif status in ('FAILED', 'ABORTED'):
                raise ValueError(f'Statement {statement_id!r} terminated with status {status}. Response details: {pformat(resp)}')
            else:
                self.log.info('Query %s', status)
            time.sleep(poll_interval)

    def get_table_primary_key(self, table: str, database: str, schema: str | None='public', cluster_identifier: str | None=None, workgroup_name: str | None=None, db_user: str | None=None, secret_arn: str | None=None, statement_name: str | None=None, with_event: bool=False, wait_for_completion: bool=True, poll_interval: int=10) -> list[str] | None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Return the table primary key.\n\n        Copied from ``RedshiftSQLHook.get_table_primary_key()``\n\n        :param table: Name of the target table\n        :param database: the name of the database\n        :param schema: Name of the target schema, public by default\n        :param sql: the SQL statement or list of  SQL statement to run\n        :param cluster_identifier: unique identifier of a cluster\n        :param db_user: the database username\n        :param secret_arn: the name or ARN of the secret that enables db access\n        :param statement_name: the name of the SQL statement\n        :param with_event: indicates whether to send an event to EventBridge\n        :param wait_for_completion: indicates whether to wait for a result, if True wait, if False don't wait\n        :param poll_interval: how often in seconds to check the query status\n\n        :return: Primary key columns list\n        "
        sql = f"\n            select kcu.column_name\n            from information_schema.table_constraints tco\n                    join information_schema.key_column_usage kcu\n                        on kcu.constraint_name = tco.constraint_name\n                            and kcu.constraint_schema = tco.constraint_schema\n                            and kcu.constraint_name = tco.constraint_name\n            where tco.constraint_type = 'PRIMARY KEY'\n            and kcu.table_schema = {schema}\n            and kcu.table_name = {table}\n        "
        stmt_id = self.execute_query(sql=sql, database=database, cluster_identifier=cluster_identifier, workgroup_name=workgroup_name, db_user=db_user, secret_arn=secret_arn, statement_name=statement_name, with_event=with_event, wait_for_completion=wait_for_completion, poll_interval=poll_interval)
        pk_columns = []
        token = ''
        while True:
            kwargs = {'Id': stmt_id}
            if token:
                kwargs['NextToken'] = token
            response = self.conn.get_statement_result(**kwargs)
            pk_columns += [y['stringValue'] for x in response['Records'] for y in x]
            if 'NextToken' in response:
                token = response['NextToken']
            else:
                break
        return pk_columns or None