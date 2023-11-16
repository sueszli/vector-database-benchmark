from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.models import BaseOperator
from airflow.providers.salesforce.hooks.salesforce import SalesforceHook
if TYPE_CHECKING:
    from airflow.utils.context import Context

class SalesforceApexRestOperator(BaseOperator):
    """
    Execute a APEX Rest API action.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:SalesforceApexRestOperator`

    :param endpoint: The REST endpoint for the request.
    :param method: HTTP method for the request (default GET)
    :param payload: A dict of parameters to send in a POST / PUT request
    :param salesforce_conn_id: The :ref:`Salesforce Connection id <howto/connection:SalesforceHook>`.
    """

    def __init__(self, *, endpoint: str, method: str='GET', payload: dict, salesforce_conn_id: str='salesforce_default', **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.endpoint = endpoint
        self.method = method
        self.payload = payload
        self.salesforce_conn_id = salesforce_conn_id

    def execute(self, context: Context) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Makes an HTTP request to an APEX REST endpoint and pushes results to xcom.\n\n        :param context: The task context during execution.\n        :return: Apex response\n        '
        result: dict = {}
        sf_hook = SalesforceHook(salesforce_conn_id=self.salesforce_conn_id)
        conn = sf_hook.get_conn()
        execution_result = conn.apexecute(action=self.endpoint, method=self.method, data=self.payload)
        if self.do_xcom_push:
            result = execution_result
        return result