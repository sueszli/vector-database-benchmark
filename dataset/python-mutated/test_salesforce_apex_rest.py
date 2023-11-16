from __future__ import annotations
from unittest.mock import Mock, patch
from airflow.providers.salesforce.operators.salesforce_apex_rest import SalesforceApexRestOperator

class TestSalesforceApexRestOperator:
    """
    Test class for SalesforceApexRestOperator
    """

    @patch('airflow.providers.salesforce.operators.salesforce_apex_rest.SalesforceHook.get_conn')
    def test_execute_salesforce_apex_rest(self, mock_get_conn):
        if False:
            i = 10
            return i + 15
        '\n        Test execute apex rest\n        '
        endpoint = 'User/Activity'
        method = 'POST'
        payload = {'activity': [{'user': '12345', 'action': 'update page', 'time': '2014-04-21T13:00:15Z'}]}
        mock_get_conn.return_value.apexecute = Mock()
        operator = SalesforceApexRestOperator(task_id='task', endpoint=endpoint, method=method, payload=payload)
        operator.execute(context={})
        mock_get_conn.return_value.apexecute.assert_called_once_with(action=endpoint, method=method, data=payload)