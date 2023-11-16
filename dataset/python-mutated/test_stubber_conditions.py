from botocore.stub import Stubber
import boto3
from boto3.dynamodb.conditions import Attr, Key
from tests import unittest

class TestStubberSupportsFilterExpressions(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        if False:
            while True:
                i = 10
        self.resource = boto3.resource('dynamodb', 'us-east-1')

    def test_table_query_can_be_stubbed_with_expressions(self):
        if False:
            i = 10
            return i + 15
        table = self.resource.Table('mytable')
        key_expr = Key('mykey').eq('testkey')
        filter_expr = Attr('myattr').eq('foo') & (Attr('myattr2').lte('buzz') | Attr('myattr2').gte('fizz'))
        stubber = Stubber(table.meta.client)
        stubber.add_response('query', dict(Items=list()), expected_params=dict(TableName='mytable', KeyConditionExpression=key_expr, FilterExpression=filter_expr))
        with stubber:
            response = table.query(KeyConditionExpression=key_expr, FilterExpression=filter_expr)
        assert response['Items'] == []
        stubber.assert_no_pending_responses()

    def test_table_scan_can_be_stubbed_with_expressions(self):
        if False:
            print('Hello World!')
        table = self.resource.Table('mytable')
        filter_expr = Attr('myattr').eq('foo') & (Attr('myattr2').lte('buzz') | Attr('myattr2').gte('fizz'))
        stubber = Stubber(table.meta.client)
        stubber.add_response('scan', dict(Items=list()), expected_params=dict(TableName='mytable', FilterExpression=filter_expr))
        with stubber:
            response = table.scan(FilterExpression=filter_expr)
        assert response['Items'] == []
        stubber.assert_no_pending_responses()