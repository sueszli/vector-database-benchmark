from __future__ import absolute_import
import mock
from st2tests import DbTestCase
from st2common.persistence.auth import User
from st2common.models.utils.profiling import enable_profiling
from st2common.models.utils.profiling import disable_profiling
from st2common.models.utils.profiling import log_query_and_profile_data_for_queryset

class MongoDBProfilingTestCase(DbTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(MongoDBProfilingTestCase, self).setUp()
        disable_profiling()

    @mock.patch('st2common.models.utils.profiling.LOG')
    def test_logging_profiling_is_disabled(self, mock_log):
        if False:
            print('Hello World!')
        disable_profiling()
        queryset = User.query(name__in=['test1', 'test2'], order_by=['+aa', '-bb'], limit=1)
        result = log_query_and_profile_data_for_queryset(queryset=queryset)
        self.assertEqual(queryset, result)
        call_args_list = mock_log.debug.call_args_list
        self.assertItemsEqual(call_args_list, [])

    @mock.patch('st2common.models.utils.profiling.LOG')
    def test_logging_profiling_is_enabled(self, mock_log):
        if False:
            for i in range(10):
                print('nop')
        enable_profiling()
        queryset = User.query(name__in=['test1', 'test2'], order_by=['+aa', '-bb'], limit=1)
        result = log_query_and_profile_data_for_queryset(queryset=queryset)
        call_args_list = mock_log.debug.call_args_list
        call_args = call_args_list[0][0]
        call_kwargs = call_args_list[0][1]
        expected_result = "db.user_d_b.find({'name': {'$in': ['test1', 'test2']}}).sort({aa: 1, bb: -1}).limit(1);"
        self.assertEqual(queryset, result)
        self.assertIn(expected_result, call_args[0])
        self.assertIn('mongo_query', call_kwargs['extra'])
        self.assertIn('mongo_shell_query', call_kwargs['extra'])

    def test_logging_profiling_is_enabled_non_queryset_object(self):
        if False:
            for i in range(10):
                print('nop')
        enable_profiling()
        queryset = 1
        result = log_query_and_profile_data_for_queryset(queryset)
        self.assertEqual(result, queryset)