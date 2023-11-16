from __future__ import annotations
from unittest import mock
from airflow.providers.amazon.aws.hooks.quicksight import QuickSightHook
from airflow.providers.amazon.aws.operators.quicksight import QuickSightCreateIngestionOperator
DATA_SET_ID = 'DemoDataSet'
INGESTION_ID = 'DemoDataSet_Ingestion'
AWS_ACCOUNT_ID = '123456789012'
INGESTION_TYPE = 'FULL_REFRESH'
MOCK_RESPONSE = {'Status': 201, 'Arn': 'arn:aws:quicksight:us-east-1:123456789012:dataset/DemoDataSet/ingestion/DemoDataSet_Ingestion', 'IngestionId': 'DemoDataSet_Ingestion', 'IngestionStatus': 'INITIALIZED', 'RequestId': 'fc1f7eea-1327-41d6-9af7-c12f097ed343'}

class TestQuickSightCreateIngestionOperator:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.quicksight = QuickSightCreateIngestionOperator(task_id='test_quicksight_operator', data_set_id=DATA_SET_ID, ingestion_id=INGESTION_ID)

    @mock.patch.object(QuickSightHook, 'get_conn')
    @mock.patch.object(QuickSightHook, 'create_ingestion')
    def test_execute(self, mock_create_ingestion, mock_client):
        if False:
            for i in range(10):
                print('nop')
        mock_create_ingestion.return_value = MOCK_RESPONSE
        self.quicksight.execute(None)
        mock_create_ingestion.assert_called_once_with(data_set_id=DATA_SET_ID, ingestion_id=INGESTION_ID, ingestion_type='FULL_REFRESH', wait_for_completion=True, check_interval=30)