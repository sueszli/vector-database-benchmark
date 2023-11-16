from __future__ import annotations
from unittest import mock
from airflow.providers.amazon.aws.operators.glue_crawler import GlueCrawlerOperator
mock_crawler_name = 'test-crawler'
mock_role_name = 'test-role'
mock_config = {'Name': mock_crawler_name, 'Description': 'Test glue crawler from Airflow', 'DatabaseName': 'test_db', 'Role': mock_role_name, 'Targets': {'S3Targets': [{'Path': 's3://test-glue-crawler/foo/', 'Exclusions': ['s3://test-glue-crawler/bar/'], 'ConnectionName': 'test-s3-conn'}], 'JdbcTargets': [{'ConnectionName': 'test-jdbc-conn', 'Path': 'test_db/test_table>', 'Exclusions': ['string']}], 'MongoDBTargets': [{'ConnectionName': 'test-mongo-conn', 'Path': 'test_db/test_collection', 'ScanAll': True}], 'DynamoDBTargets': [{'Path': 'test_db/test_table', 'scanAll': True, 'scanRate': 123.0}], 'CatalogTargets': [{'DatabaseName': 'test_glue_db', 'Tables': ['test']}]}, 'Classifiers': ['test-classifier'], 'TablePrefix': 'test', 'SchemaChangePolicy': {'UpdateBehavior': 'UPDATE_IN_DATABASE', 'DeleteBehavior': 'DEPRECATE_IN_DATABASE'}, 'RecrawlPolicy': {'RecrawlBehavior': 'CRAWL_EVERYTHING'}, 'LineageConfiguration': 'ENABLE', 'Configuration': '\n    {\n        "Version": 1.0,\n        "CrawlerOutput": {\n            "Partitions": { "AddOrUpdateBehavior": "InheritFromTable" }\n        }\n    }\n    ', 'SecurityConfiguration': 'test', 'Tags': {'test': 'foo'}}

class TestGlueCrawlerOperator:

    def setup_method(self):
        if False:
            return 10
        self.glue = GlueCrawlerOperator(task_id='test_glue_crawler_operator', config=mock_config)

    @mock.patch('airflow.providers.amazon.aws.operators.glue_crawler.GlueCrawlerHook')
    def test_execute_without_failure(self, mock_hook):
        if False:
            return 10
        mock_hook.return_value.has_crawler.return_value = True
        self.glue.execute({})
        mock_hook.assert_has_calls([mock.call('aws_default', region_name=None), mock.call().has_crawler('test-crawler'), mock.call().update_crawler(**mock_config), mock.call().start_crawler(mock_crawler_name), mock.call().wait_for_crawler_completion(crawler_name=mock_crawler_name, poll_interval=5)])