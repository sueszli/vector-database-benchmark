"""st.cache_data unit tests."""
from __future__ import annotations
import logging
import os
import pickle
import re
import threading
import unittest
from typing import Any
from unittest.mock import MagicMock, Mock, mock_open, patch
from parameterized import parameterized
import streamlit as st
from streamlit import file_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Text_pb2 import Text as TextProto
from streamlit.runtime import Runtime
from streamlit.runtime.caching import cache_data_api
from streamlit.runtime.caching.cache_data_api import get_data_cache_stats_provider
from streamlit.runtime.caching.cache_errors import CacheError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cached_message_replay import CachedResult, ElementMsgData, MultiCacheResults, _make_widget_key
from streamlit.runtime.caching.hashing import UserHashError
from streamlit.runtime.caching.storage import CacheStorage, CacheStorageContext, CacheStorageManager
from streamlit.runtime.caching.storage.cache_storage_protocol import InvalidCacheStorageContext
from streamlit.runtime.caching.storage.dummy_cache_storage import DummyCacheStorage, MemoryCacheStorageManager
from streamlit.runtime.caching.storage.local_disk_cache_storage import LocalDiskCacheStorageManager, get_cache_folder_path
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.stats import CacheStat
from tests.delta_generator_test_case import DeltaGeneratorTestCase
from tests.streamlit.runtime.caching.common_cache_test import as_cached_result as _as_cached_result
from tests.testutil import create_mock_script_run_ctx

def as_cached_result(value: Any) -> MultiCacheResults:
    if False:
        return 10
    return _as_cached_result(value, CacheType.DATA)

def as_replay_test_data() -> MultiCacheResults:
    if False:
        print('Hello World!')
    'Creates cached results for a function that returned 1\n    and executed `st.text(1)`.\n    '
    widget_key = _make_widget_key([], CacheType.DATA)
    d = {}
    d[widget_key] = CachedResult(1, [ElementMsgData('text', TextProto(body='1'), st._main.id, '')], st._main.id, st.sidebar.id)
    return MultiCacheResults(set(), d)

class CacheDataTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        add_script_run_ctx(threading.current_thread(), create_mock_script_run_ctx())
        mock_runtime = MagicMock(spec=Runtime)
        mock_runtime.cache_storage_manager = MemoryCacheStorageManager()
        Runtime._instance = mock_runtime

    def tearDown(self):
        if False:
            print('Hello World!')
        cache_data_api.CACHE_DATA_MESSAGE_REPLAY_CTX._cached_func_stack = []
        cache_data_api.CACHE_DATA_MESSAGE_REPLAY_CTX._suppress_st_function_warning = 0
        st.cache_data.clear()

    @patch.object(st, 'exception')
    def test_mutate_return(self, exception):
        if False:
            return 10
        "Mutating a memoized return value is legal, and *won't* affect\n        future accessors of the data."

        @st.cache_data
        def f():
            if False:
                for i in range(10):
                    print('nop')
            return [0, 1]
        r1 = f()
        r1[0] = 1
        r2 = f()
        exception.assert_not_called()
        self.assertEqual(r1, [1, 1])
        self.assertEqual(r2, [0, 1])

    def test_multiple_api_names(self):
        if False:
            return 10
        '`st.experimental_memo` is effectively an alias for `st.cache_data`, and we\n        support both APIs while experimental_memo is being deprecated.\n        '
        num_calls = [0]

        def foo():
            if False:
                return 10
            num_calls[0] += 1
            return 42
        cache_data_func = st.cache_data(foo)
        memo_func = st.experimental_memo(foo)
        self.assertEqual(42, cache_data_func())
        self.assertEqual(42, memo_func())
        self.assertEqual(1, num_calls[0])

    @parameterized.expand([('cache_data', st.cache_data, False), ('experimental_memo', st.experimental_memo, True)])
    @patch('streamlit.runtime.caching.cache_data_api.show_deprecation_warning')
    def test_deprecation_warnings(self, _, decorator: Any, should_show_warning: bool, show_warning_mock: Mock):
        if False:
            return 10
        'We show deprecation warnings when using `@st.experimental_memo`, but not `@st.cache_data`.'
        warning_str = '`st.experimental_memo` is deprecated. Please use the new command `st.cache_data` instead, which has the same behavior. More information [in our docs](https://docs.streamlit.io/library/advanced-features/caching).'

        @decorator
        def foo():
            if False:
                while True:
                    i = 10
            return 42
        if should_show_warning:
            show_warning_mock.assert_called_once_with(warning_str)
        else:
            show_warning_mock.assert_not_called()
        show_warning_mock.reset_mock()
        decorator.clear()
        if should_show_warning:
            show_warning_mock.assert_called_once_with(warning_str)
        else:
            show_warning_mock.assert_not_called()

    def test_cached_member_function_with_hash_func(self):
        if False:
            for i in range(10):
                print('nop')
        '@st.cache_data can be applied to class member functions\n        with corresponding hash_func.\n        '

        class TestClass:

            @st.cache_data(hash_funcs={'tests.streamlit.runtime.caching.cache_data_api_test.CacheDataTest.test_cached_member_function_with_hash_func.<locals>.TestClass': id})
            def member_func(self):
                if False:
                    i = 10
                    return i + 15
                return 'member func!'

            @classmethod
            @st.cache_data
            def class_method(cls):
                if False:
                    i = 10
                    return i + 15
                return 'class method!'

            @staticmethod
            @st.cache_data
            def static_method():
                if False:
                    for i in range(10):
                        print('nop')
                return 'static method!'
        obj = TestClass()
        self.assertEqual('member func!', obj.member_func())
        self.assertEqual('class method!', obj.class_method())
        self.assertEqual('static method!', obj.static_method())

    def test_function_name_does_not_use_hashfuncs(self):
        if False:
            return 10
        "Hash funcs should only be used on arguments to a function,\n        and not when computing the key for a function's unique MemCache.\n        "
        str_hash_func = Mock(return_value=None)

        @st.cache_data(hash_funcs={str: str_hash_func})
        def foo(string_arg):
            if False:
                while True:
                    i = 10
            return []
        foo('ahoy')
        str_hash_func.assert_called_once_with('ahoy')

    def test_user_hash_error(self):
        if False:
            return 10

        class MyObj:

            def __repr__(self):
                if False:
                    i = 10
                    return i + 15
                return 'MyObj class'

        def bad_hash_func(x):
            if False:
                i = 10
                return i + 15
            x += 10
            return x

        @st.cache_data(hash_funcs={MyObj: bad_hash_func})
        def user_hash_error_func(x):
            if False:
                return 10
            pass
        with self.assertRaises(UserHashError) as ctx:
            my_obj = MyObj()
            user_hash_error_func(my_obj)
        expected_message = "unsupported operand type(s) for +=: 'MyObj' and 'int'\n\nThis error is likely due to a bug in `bad_hash_func()`, which is a\nuser-defined hash function that was passed into the `@st.cache_data` decorator of\n`user_hash_error_func()`.\n\n`bad_hash_func()` failed when hashing an object of type\n`tests.streamlit.runtime.caching.cache_data_api_test.CacheDataTest.test_user_hash_error.<locals>.MyObj`.  If you don't know where that object is coming from,\ntry looking at the hash chain below for an object that you do recognize, then\npass that to `hash_funcs` instead:\n\n```\nObject of type tests.streamlit.runtime.caching.cache_data_api_test.CacheDataTest.test_user_hash_error.<locals>.MyObj: MyObj class\n```\n\nIf you think this is actually a Streamlit bug, please\n[file a bug report here](https://github.com/streamlit/streamlit/issues/new/choose)."
        self.assertEqual(str(ctx.exception), expected_message)

class CacheDataPersistTest(DeltaGeneratorTestCase):
    """st.cache_data disk persistence tests"""

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        mock_runtime = MagicMock(spec=Runtime)
        mock_runtime.cache_storage_manager = LocalDiskCacheStorageManager()
        Runtime._instance = mock_runtime

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        st.cache_data.clear()
        super().tearDown()

    @patch('streamlit.runtime.caching.storage.local_disk_cache_storage.streamlit_write')
    def test_dont_persist_by_default(self, mock_write):
        if False:
            for i in range(10):
                print('nop')

        @st.cache_data
        def foo():
            if False:
                while True:
                    i = 10
            return 'data'
        foo()
        mock_write.assert_not_called()

    @patch('streamlit.runtime.caching.storage.local_disk_cache_storage.streamlit_write')
    def test_persist_path(self, mock_write):
        if False:
            return 10
        "Ensure we're writing to ~/.streamlit/cache/*.memo"

        @st.cache_data(persist='disk')
        def foo():
            if False:
                print('Hello World!')
            return 'data'
        foo()
        mock_write.assert_called_once()
        write_path = mock_write.call_args[0][0]
        match = re.fullmatch('/mock/home/folder/.streamlit/cache/.*?\\.memo', write_path)
        self.assertIsNotNone(match)

    @patch('streamlit.file_util.os.stat', MagicMock())
    @patch('streamlit.file_util.open', mock_open(read_data=pickle.dumps(as_cached_result('mock_pickled_value'))))
    @patch('streamlit.runtime.caching.storage.local_disk_cache_storage.streamlit_read', wraps=file_util.streamlit_read)
    def test_read_persisted_data(self, mock_read):
        if False:
            return 10
        'We should read persisted data from disk on cache miss.'

        @st.cache_data(persist='disk')
        def foo():
            if False:
                return 10
            return 'actual_value'
        data = foo()
        mock_read.assert_called_once()
        self.assertEqual('mock_pickled_value', data)

    @patch('streamlit.file_util.os.stat', MagicMock())
    @patch('streamlit.file_util.open', mock_open(read_data='bad_pickled_value'))
    @patch('streamlit.runtime.caching.storage.local_disk_cache_storage.streamlit_read', wraps=file_util.streamlit_read)
    def test_read_bad_persisted_data(self, mock_read):
        if False:
            return 10
        'If our persisted data is bad, we raise an exception.'

        @st.cache_data(persist='disk')
        def foo():
            if False:
                while True:
                    i = 10
            return 'actual_value'
        with self.assertRaises(CacheError) as error:
            foo()
        mock_read.assert_called_once()
        self.assertEqual('Unable to read from cache', str(error.exception))

    @patch('streamlit.file_util.os.stat', MagicMock())
    @patch('streamlit.file_util.open', mock_open(read_data=b'bad_binary_pickled_value'))
    @patch('streamlit.runtime.caching.storage.local_disk_cache_storage.streamlit_read', wraps=file_util.streamlit_read)
    def test_read_bad_persisted_binary_data(self, mock_read):
        if False:
            return 10
        'If our persisted data is bad, we raise an exception.'

        @st.cache_data(persist='disk')
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            return 'actual_value'
        with self.assertRaises(CacheError) as error:
            foo()
        mock_read.assert_called_once()
        self.assertIn('Failed to unpickle', str(error.exception))

    def test_bad_persist_value(self):
        if False:
            print('Hello World!')
        "Throw an error if an invalid value is passed to 'persist'."
        with self.assertRaises(StreamlitAPIException) as e:

            @st.cache_data(persist='yesplz')
            def foo():
                if False:
                    while True:
                        i = 10
                pass
        self.assertEqual("Unsupported persist option 'yesplz'. Valid values are 'disk' or None.", str(e.exception))

    @patch('shutil.rmtree')
    def test_clear_all_disk_caches(self, mock_rmtree):
        if False:
            return 10
        '`clear_all` should remove the disk cache directory if it exists.'
        with patch('os.path.isdir', MagicMock(return_value=True)):
            st.cache_data.clear()
            mock_rmtree.assert_called_once_with(get_cache_folder_path())
        mock_rmtree.reset_mock()
        with patch('os.path.isdir', MagicMock(return_value=False)):
            st.cache_data.clear()
            mock_rmtree.assert_not_called()

    @patch('streamlit.file_util.os.stat', MagicMock())
    @patch('streamlit.file_util.open', wraps=mock_open(read_data=pickle.dumps(as_cached_result('mock_pickled_value'))))
    @patch('streamlit.runtime.caching.storage.local_disk_cache_storage.os.remove')
    def test_clear_one_disk_cache(self, mock_os_remove: Mock, mock_open: Mock):
        if False:
            return 10
        "A memoized function's clear_cache() property should just clear\n        that function's cache."

        @st.cache_data(persist='disk')
        def foo(val):
            if False:
                return 10
            return 'actual_value'
        foo(0)
        foo(1)
        self.assertEqual(2, mock_open.call_count)
        created_filenames = {mock_open.call_args_list[0][0][0], mock_open.call_args_list[1][0][0]}
        created_files_base_names = [os.path.basename(filename) for filename in created_filenames]
        mock_os_remove.assert_not_called()
        with patch('os.listdir', MagicMock(return_value=created_files_base_names)), patch('os.path.isdir', MagicMock(return_value=True)):
            foo.clear()
        self.assertEqual(2, mock_os_remove.call_count)
        removed_filenames = {mock_os_remove.call_args_list[0][0][0], mock_os_remove.call_args_list[1][0][0]}
        self.assertEqual(created_filenames, removed_filenames)

    @patch('streamlit.file_util.os.stat', MagicMock())
    @patch('streamlit.file_util.open', wraps=mock_open(read_data=pickle.dumps(as_replay_test_data())))
    def test_cached_st_function_replay(self, _):
        if False:
            print('Hello World!')

        @st.cache_data(persist='disk')
        def foo(i):
            if False:
                i = 10
                return i + 15
            st.text(i)
            return i
        foo(1)
        deltas = self.get_all_deltas_from_queue()
        text = [element.text.body for element in (delta.new_element for delta in deltas) if element.WhichOneof('type') == 'text']
        assert text == ['1']

    @patch('streamlit.file_util.os.stat', MagicMock())
    @patch('streamlit.runtime.caching.storage.local_disk_cache_storage.streamlit_write', MagicMock())
    @patch('streamlit.file_util.open', wraps=mock_open(read_data=pickle.dumps(1)))
    def test_cached_format_migration(self, _):
        if False:
            return 10

        @st.cache_data(persist='disk')
        def foo(i):
            if False:
                print('Hello World!')
            st.text(i)
            return i
        foo(1)

    @patch('streamlit.runtime.caching.storage.local_disk_cache_storage.streamlit_write')
    def test_warning_memo_ttl_persist(self, _):
        if False:
            for i in range(10):
                print('nop')
        'Using @st.cache_data with ttl and persist produces a warning.'
        with self.assertLogs('streamlit.runtime.caching.storage.local_disk_cache_storage', level=logging.WARNING) as logs:

            @st.cache_data(ttl=60, persist='disk')
            def user_function():
                if False:
                    while True:
                        i = 10
                return 42
            st.write(user_function())
            output = ''.join(logs.output)
            self.assertIn("The cached function 'user_function' has a TTL that will be ignored.", output)

    @parameterized.expand([('disk', 'disk', True), ('True', True, True), ('None', None, False), ('False', False, False)])
    @patch('streamlit.runtime.caching.storage.local_disk_cache_storage.streamlit_write')
    def test_persist_param_value(self, _, persist_value: str | bool | None, should_persist: bool, mock_write: Mock):
        if False:
            for i in range(10):
                print('nop')
        'Passing "disk" or `True` enables persistence; `None` or `False` disables it.'

        @st.cache_data(persist=persist_value)
        def foo():
            if False:
                print('Hello World!')
            return 'data'
        foo()
        if should_persist:
            mock_write.assert_called_once()
        else:
            mock_write.assert_not_called()

class CacheDataStatsProviderTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        add_script_run_ctx(threading.current_thread(), create_mock_script_run_ctx())
        mock_runtime = MagicMock(spec=Runtime)
        mock_runtime.cache_storage_manager = MemoryCacheStorageManager()
        Runtime._instance = mock_runtime
        st.cache_data.clear()

    def tearDown(self):
        if False:
            return 10
        st.cache_data.clear()

    def test_no_stats(self):
        if False:
            while True:
                i = 10
        self.assertEqual([], get_data_cache_stats_provider().get_stats())

    def test_multiple_stats(self):
        if False:
            return 10

        @st.cache_data
        def foo(count):
            if False:
                return 10
            return [3.14] * count

        @st.cache_data
        def bar():
            if False:
                while True:
                    i = 10
            return 'shivermetimbers'
        foo(1)
        foo(53)
        bar()
        bar()
        foo_cache_name = f'{foo.__module__}.{foo.__qualname__}'
        bar_cache_name = f'{bar.__module__}.{bar.__qualname__}'
        expected = [CacheStat(category_name='st_cache_data', cache_name=foo_cache_name, byte_length=get_byte_length(as_cached_result([3.14]))), CacheStat(category_name='st_cache_data', cache_name=foo_cache_name, byte_length=get_byte_length(as_cached_result([3.14] * 53))), CacheStat(category_name='st_cache_data', cache_name=bar_cache_name, byte_length=get_byte_length(as_cached_result('shivermetimbers')))]
        self.assertEqual(set(expected), set(get_data_cache_stats_provider().get_stats()))

class CacheDataValidateParamsTest(DeltaGeneratorTestCase):
    """st.cache_data disk persistence tests"""

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        mock_runtime = MagicMock(spec=Runtime)
        mock_runtime.cache_storage_manager = AlwaysFailingTestCacheStorageManager()
        Runtime._instance = mock_runtime

    def test_error_logged_and_raised_on_improperly_configured_cache_data(self):
        if False:
            print('Hello World!')
        with self.assertRaises(InvalidCacheStorageContext) as e, self.assertLogs('streamlit.runtime.caching.cache_data_api', level=logging.ERROR) as logs:

            @st.cache_data(persist='disk')
            def foo():
                if False:
                    while True:
                        i = 10
                return 'data'
        self.assertEqual(str(e.exception), 'This CacheStorageManager always fails')
        output = ''.join(logs.output)
        self.assertIn('This CacheStorageManager always fails', output)

def get_byte_length(value):
    if False:
        print('Hello World!')
    'Return the byte length of the pickled value.'
    return len(pickle.dumps(value))

class AlwaysFailingTestCacheStorageManager(CacheStorageManager):
    """A CacheStorageManager that always fails in check_context."""

    def create(self, context: CacheStorageContext) -> CacheStorage:
        if False:
            print('Hello World!')
        return DummyCacheStorage()

    def clear_all(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def check_context(self, context: CacheStorageContext) -> None:
        if False:
            while True:
                i = 10
        raise InvalidCacheStorageContext('This CacheStorageManager always fails')