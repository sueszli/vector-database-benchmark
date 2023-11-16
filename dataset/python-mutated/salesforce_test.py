"""
Unit test for the Salesforce contrib package
"""
from luigi.contrib.salesforce import SalesforceAPI, QuerySalesforce
from helpers import unittest
import mock
from luigi.mock import MockTarget
import re
import pytest

def mocked_requests_get(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')

    class MockResponse:

        def __init__(self, body, status_code):
            if False:
                while True:
                    i = 10
            self.body = body
            self.status_code = status_code

        @property
        def text(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.body

        def raise_for_status(self):
            if False:
                i = 10
                return i + 15
            return None
    result_list = '<result-list xmlns="http://www.force.com/2009/06/asyncapi/dataload"><result>1234</result><result>1235</result><result>1236</result></result-list>'
    return MockResponse(result_list, 200)
old__open = open

def mocked_open(*args, **kwargs):
    if False:
        print('Hello World!')
    if re.match('job_data', str(args[0])):
        return MockTarget(args[0]).open(args[1])
    else:
        return old__open(*args)

@pytest.mark.contrib
class TestSalesforceAPI(unittest.TestCase):

    @mock.patch('requests.get', side_effect=mocked_requests_get)
    def test_deprecated_results_warning(self, mock_get):
        if False:
            while True:
                i = 10
        sf = SalesforceAPI('xx', 'xx', 'xx')
        with self.assertWarnsRegex(UserWarning, 'get_batch_results is deprecated'):
            result_id = sf.get_batch_results('job_id', 'batch_id')
            self.assertEqual('1234', result_id)

    @mock.patch('requests.get', side_effect=mocked_requests_get)
    def test_result_ids(self, mock_get):
        if False:
            for i in range(10):
                print('nop')
        sf = SalesforceAPI('xx', 'xx', 'xx')
        result_ids = sf.get_batch_result_ids('job_id', 'batch_id')
        self.assertEqual(['1234', '1235', '1236'], result_ids)

class TestQuerySalesforce(QuerySalesforce):

    def output(self):
        if False:
            return 10
        return MockTarget('job_data.csv')

    @property
    def object_name(self):
        if False:
            return 10
        return 'dual'

    @property
    def soql(self):
        if False:
            while True:
                i = 10
        return 'SELECT * FROM %s' % self.object_name

@pytest.mark.contrib
class TestSalesforceQuery(unittest.TestCase):

    @mock.patch('builtins.open', side_effect=mocked_open)
    def setUp(self, mock_open):
        if False:
            return 10
        MockTarget.fs.clear()
        self.result_ids = ['a', 'b', 'c']
        counter = 1
        self.all_lines = 'Lines\n'
        self.header = 'Lines'
        for (i, id) in enumerate(self.result_ids):
            filename = '%s.%d' % ('job_data.csv', i)
            with MockTarget(filename).open('w') as f:
                line = '%d line\n%d line' % (counter, counter + 1)
                f.write(self.header + '\n' + line + '\n')
                self.all_lines += line + '\n'
                counter += 2

    @mock.patch('builtins.open', side_effect=mocked_open)
    def test_multi_csv_download(self, mock_open):
        if False:
            for i in range(10):
                print('nop')
        qsf = TestQuerySalesforce()
        qsf.merge_batch_results(self.result_ids)
        self.assertEqual(MockTarget(qsf.output().path).open('r').read(), self.all_lines)