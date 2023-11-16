import boto3
import mock
import os
import unittest
try:
    from mock import patch
except ImportError:
    from unittest.mock import patch
zappa_async = __import__('zappa.async', fromlist=['AsyncException', 'LambdaAsyncResponse', 'SnsAsyncResponse', 'import_and_get_task', 'get_func_task_path'])
AsyncException = zappa_async.AsyncException
LambdaAsyncResponse = zappa_async.LambdaAsyncResponse
SnsAsyncResponse = zappa_async.SnsAsyncResponse
import_and_get_task = zappa_async.import_and_get_task
get_func_task_path = zappa_async.get_func_task_path

class TestZappa(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.sleep_patch = mock.patch('time.sleep', return_value=None)
        self.users_current_region_name = os.environ.get('AWS_DEFAULT_REGION', None)
        os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
        if not os.environ.get('PLACEBO_MODE') == 'record':
            self.sleep_patch.start()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if not os.environ.get('PLACEBO_MODE') == 'record':
            self.sleep_patch.stop()
        del os.environ['AWS_DEFAULT_REGION']
        if self.users_current_region_name is not None:
            os.environ['AWS_DEFAULT_REGION'] = self.users_current_region_name

    def test_test(self):
        if False:
            return 10
        self.assertTrue(True)
        self.assertFalse(False)

    def test_nofails_classes(self):
        if False:
            for i in range(10):
                print('nop')
        boto_session = boto3.Session(region_name=os.environ['AWS_DEFAULT_REGION'])
        a = AsyncException()
        l = LambdaAsyncResponse(boto_session=boto_session)
        s = SnsAsyncResponse(arn='arn:abc:def', boto_session=boto_session)

    def test_nofails_funcs(self):
        if False:
            for i in range(10):
                print('nop')
        funk = import_and_get_task('tests.test_app.async_me')
        get_func_task_path(funk)
        self.assertEqual(funk.__name__, 'async_me')

    def test_sync_call(self):
        if False:
            for i in range(10):
                print('nop')
        funk = import_and_get_task('tests.test_app.async_me')
        self.assertEqual(funk.sync('123'), 'run async when on lambda 123')

    def test_async_call_with_defaults(self):
        if False:
            i = 10
            return i + 15
        "Change a task's asynchronousity at runtime."
        async_me = import_and_get_task('tests.test_app.async_me')
        lambda_async_mock = mock.Mock()
        lambda_async_mock.return_value.send.return_value = 'Running async!'
        with mock.patch.dict('zappa.async.ASYNC_CLASSES', {'lambda': lambda_async_mock}):
            self.assertEqual(async_me('123'), 'run async when on lambda 123')
            options = {'AWS_LAMBDA_FUNCTION_NAME': 'MyLambda', 'AWS_REGION': 'us-east-1'}
            with mock.patch.dict(os.environ, options):
                self.assertEqual(async_me('qux'), 'Running async!')
        lambda_async_mock.assert_called_once()
        lambda_async_mock.assert_called_with(aws_region='us-east-1', capture_response=False, lambda_function_name='MyLambda')
        lambda_async_mock.return_value.send.assert_called_with(get_func_task_path(async_me), ('qux',), {})