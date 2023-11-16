"""Hook for JIRA."""
from __future__ import annotations
from typing import Any
from atlassian import Jira
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook

class JiraHook(BaseHook):
    """
    Jira interaction hook, a Wrapper around Atlassian Jira Python SDK.

    :param jira_conn_id: reference to a pre-defined Jira Connection
    """
    default_conn_name = 'jira_default'
    conn_type = 'jira'
    conn_name_attr = 'jira_conn_id'
    hook_name = 'JIRA'

    def __init__(self, jira_conn_id: str=default_conn_name, proxies: Any | None=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.jira_conn_id = jira_conn_id
        self.proxies = proxies
        self.client: Jira | None = None
        self.get_conn()

    def get_conn(self) -> Jira:
        if False:
            while True:
                i = 10
        if not self.client:
            self.log.debug('Creating Jira client for conn_id: %s', self.jira_conn_id)
            verify = True
            if not self.jira_conn_id:
                raise AirflowException('Failed to create jira client. no jira_conn_id provided')
            conn = self.get_connection(self.jira_conn_id)
            if conn.extra is not None:
                extra_options = conn.extra_dejson
                if 'verify' in extra_options and extra_options['verify'].lower() == 'false':
                    verify = False
            self.client = Jira(url=conn.host, username=conn.login, password=conn.password, verify_ssl=verify, proxies=self.proxies)
        return self.client