from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.models import BaseOperator
if TYPE_CHECKING:
    import pandas as pd
    from slack_sdk.http_retry import RetryHandler
    from airflow.providers.common.sql.hooks.sql import DbApiHook

class BaseSqlToSlackOperator(BaseOperator):
    """
    Operator implements base sql methods for SQL to Slack Transfer operators.

    :param sql: The SQL query to be executed
    :param sql_conn_id: reference to a specific DB-API Connection.
    :param sql_hook_params: Extra config params to be passed to the underlying hook.
        Should match the desired hook constructor params.
    :param parameters: The parameters to pass to the SQL query.
    :param slack_proxy: Proxy to make the Slack Incoming Webhook / API calls. Optional
    :param slack_timeout: The maximum number of seconds the client will wait to connect
        and receive a response from Slack. Optional
    :param slack_retry_handlers: List of handlers to customize retry logic. Optional
    """

    def __init__(self, *, sql: str, sql_conn_id: str, sql_hook_params: dict | None=None, parameters: list | tuple | Mapping[str, Any] | None=None, slack_proxy: str | None=None, slack_timeout: int | None=None, slack_retry_handlers: list[RetryHandler] | None=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.sql_conn_id = sql_conn_id
        self.sql_hook_params = sql_hook_params
        self.sql = sql
        self.parameters = parameters
        self.slack_proxy = slack_proxy
        self.slack_timeout = slack_timeout
        self.slack_retry_handlers = slack_retry_handlers

    def _get_hook(self) -> DbApiHook:
        if False:
            while True:
                i = 10
        self.log.debug('Get connection for %s', self.sql_conn_id)
        conn = BaseHook.get_connection(self.sql_conn_id)
        hook = conn.get_hook(hook_params=self.sql_hook_params)
        if not callable(getattr(hook, 'get_pandas_df', None)):
            raise AirflowException('This hook is not supported. The hook class must have get_pandas_df method.')
        return hook

    def _get_query_results(self) -> pd.DataFrame:
        if False:
            return 10
        sql_hook = self._get_hook()
        self.log.info('Running SQL query: %s', self.sql)
        df = sql_hook.get_pandas_df(self.sql, parameters=self.parameters)
        return df