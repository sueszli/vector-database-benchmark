"""st.cache_resource unit tests."""
import threading
import unittest
from typing import Any, List
from unittest.mock import Mock, patch
from parameterized import parameterized
import streamlit as st
from streamlit.runtime.caching import cache_resource_api, get_resource_cache_stats_provider
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cached_message_replay import MultiCacheResults
from streamlit.runtime.caching.hashing import UserHashError
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.stats import CacheStat
from streamlit.vendor.pympler.asizeof import asizeof
from tests.streamlit.runtime.caching.common_cache_test import as_cached_result as _as_cached_result
from tests.testutil import create_mock_script_run_ctx

def as_cached_result(value: Any) -> MultiCacheResults:
    if False:
        i = 10
        return i + 15
    return _as_cached_result(value, CacheType.RESOURCE)

class CacheResourceTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        add_script_run_ctx(threading.current_thread(), create_mock_script_run_ctx())

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        st.cache_resource.clear()
        cache_resource_api.CACHE_RESOURCE_MESSAGE_REPLAY_CTX._cached_func_stack = []
        cache_resource_api.CACHE_RESOURCE_MESSAGE_REPLAY_CTX._suppress_st_function_warning = 0

    @patch.object(st, 'exception')
    def test_mutate_return(self, exception):
        if False:
            print('Hello World!')
        'Mutating a cache_resource return value is legal, and *will* affect\n        future accessors of the data.'

        @st.cache_resource
        def f():
            if False:
                while True:
                    i = 10
            return [0, 1]
        r1 = f()
        r1[0] = 1
        r2 = f()
        exception.assert_not_called()
        self.assertEqual(r1, [1, 1])
        self.assertEqual(r2, [1, 1])

    def test_multiple_api_names(self):
        if False:
            i = 10
            return i + 15
        '`st.experimental_singleton` is effectively an alias for `st.cache_resource`, and we\n        support both APIs while experimental_singleton is being deprecated.\n        '
        num_calls = [0]

        def foo():
            if False:
                print('Hello World!')
            num_calls[0] += 1
            return 42
        cache_resource_func = st.cache_resource(foo)
        singleton_func = st.experimental_singleton(foo)
        self.assertEqual(42, cache_resource_func())
        self.assertEqual(42, singleton_func())
        self.assertEqual(1, num_calls[0])

    @parameterized.expand([('cache_resource', st.cache_resource, False), ('experimental_singleton', st.experimental_singleton, True)])
    @patch('streamlit.runtime.caching.cache_resource_api.show_deprecation_warning')
    def test_deprecation_warnings(self, _, decorator: Any, should_show_warning: bool, show_warning_mock: Mock):
        if False:
            i = 10
            return i + 15
        'We show deprecation warnings when using `@st.experimental_singleton`, but not `@st.cache_resource`.'
        warning_str = '`st.experimental_singleton` is deprecated. Please use the new command `st.cache_resource` instead, which has the same behavior. More information [in our docs](https://docs.streamlit.io/library/advanced-features/caching).'

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
        '@st.cache_resource can be applied to class member functions\n        with corresponding hash_func.\n        '

        class TestClass:

            @st.cache_resource(hash_funcs={'tests.streamlit.runtime.caching.cache_resource_api_test.CacheResourceTest.test_cached_member_function_with_hash_func.<locals>.TestClass': id})
            def member_func(self):
                if False:
                    i = 10
                    return i + 15
                return 'member func!'

            @classmethod
            @st.cache_resource
            def class_method(cls):
                if False:
                    return 10
                return 'class method!'

            @staticmethod
            @st.cache_resource
            def static_method():
                if False:
                    i = 10
                    return i + 15
                return 'static method!'
        obj = TestClass()
        self.assertEqual('member func!', obj.member_func())
        self.assertEqual('class method!', obj.class_method())
        self.assertEqual('static method!', obj.static_method())

    def test_function_name_does_not_use_hashfuncs(self):
        if False:
            i = 10
            return i + 15
        "Hash funcs should only be used on arguments to a function,\n        and not when computing the key for a function's unique MemCache.\n        "
        str_hash_func = Mock(return_value=None)

        @st.cache(hash_funcs={str: str_hash_func})
        def foo(string_arg):
            if False:
                print('Hello World!')
            return []
        foo('ahoy')
        str_hash_func.assert_called_once_with('ahoy')

    def test_user_hash_error(self):
        if False:
            return 10

        class MyObj:

            def __repr__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 'MyObj class'

        def bad_hash_func(x):
            if False:
                print('Hello World!')
            x += 10
            return x

        @st.cache_resource(hash_funcs={MyObj: bad_hash_func})
        def user_hash_error_func(x):
            if False:
                print('Hello World!')
            pass
        with self.assertRaises(UserHashError) as ctx:
            my_obj = MyObj()
            user_hash_error_func(my_obj)
        expected_message = "unsupported operand type(s) for +=: 'MyObj' and 'int'\n\nThis error is likely due to a bug in `bad_hash_func()`, which is a\nuser-defined hash function that was passed into the `@st.cache_resource` decorator of\n`user_hash_error_func()`.\n\n`bad_hash_func()` failed when hashing an object of type\n`tests.streamlit.runtime.caching.cache_resource_api_test.CacheResourceTest.test_user_hash_error.<locals>.MyObj`.  If you don't know where that object is coming from,\ntry looking at the hash chain below for an object that you do recognize, then\npass that to `hash_funcs` instead:\n\n```\nObject of type tests.streamlit.runtime.caching.cache_resource_api_test.CacheResourceTest.test_user_hash_error.<locals>.MyObj: MyObj class\n```\n\nIf you think this is actually a Streamlit bug, please\n[file a bug report here](https://github.com/streamlit/streamlit/issues/new/choose)."
        self.assertEqual(str(ctx.exception), expected_message)

class CacheResourceValidateTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        add_script_run_ctx(threading.current_thread(), create_mock_script_run_ctx())

    def tearDown(self):
        if False:
            return 10
        st.cache_resource.clear()
        cache_resource_api.CACHE_RESOURCE_MESSAGE_REPLAY_CTX._cached_func_stack = []
        cache_resource_api.CACHE_RESOURCE_MESSAGE_REPLAY_CTX._suppress_st_function_warning = 0

    def test_validate_success(self):
        if False:
            print('Hello World!')
        "If we have a validate function and it returns True, we don't recompute our cached value."
        validate = Mock(return_value=True)
        call_count: List[int] = [0]

        @st.cache_resource(validate=validate)
        def f() -> int:
            if False:
                while True:
                    i = 10
            call_count[0] += 1
            return call_count[0]
        self.assertEqual(1, f())
        validate.assert_not_called()
        for _ in range(3):
            self.assertEqual(1, f())
            validate.assert_called_once_with(1)
            validate.reset_mock()

    def test_validate_fail(self):
        if False:
            return 10
        'If we have a validate function and it returns False, we recompute our cached value.'
        validate = Mock(return_value=False)
        call_count: List[int] = [0]

        @st.cache_resource(validate=validate)
        def f() -> int:
            if False:
                while True:
                    i = 10
            call_count[0] += 1
            return call_count[0]
        expected_call_count = 1
        self.assertEqual(expected_call_count, f())
        validate.assert_not_called()
        for _ in range(3):
            expected_call_count += 1
            self.assertEqual(expected_call_count, f())
            validate.assert_called_once_with(expected_call_count - 1)
            validate.reset_mock()

class CacheResourceStatsProviderTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        st.cache_resource.clear()
        add_script_run_ctx(threading.current_thread(), create_mock_script_run_ctx())

    def tearDown(self):
        if False:
            return 10
        st.cache_resource.clear()

    def test_no_stats(self):
        if False:
            while True:
                i = 10
        self.assertEqual([], get_resource_cache_stats_provider().get_stats())

    def test_multiple_stats(self):
        if False:
            return 10

        @st.cache_resource
        def foo(count):
            if False:
                print('Hello World!')
            return [3.14] * count

        @st.cache_resource
        def bar():
            if False:
                for i in range(10):
                    print('nop')
            return threading.Lock()
        foo(1)
        foo(53)
        bar()
        bar()
        foo_cache_name = f'{foo.__module__}.{foo.__qualname__}'
        bar_cache_name = f'{bar.__module__}.{bar.__qualname__}'
        expected = [CacheStat(category_name='st_cache_resource', cache_name=foo_cache_name, byte_length=get_byte_length(as_cached_result([3.14]))), CacheStat(category_name='st_cache_resource', cache_name=foo_cache_name, byte_length=get_byte_length(as_cached_result([3.14] * 53))), CacheStat(category_name='st_cache_resource', cache_name=bar_cache_name, byte_length=get_byte_length(as_cached_result(bar())))]
        self.assertEqual(set(expected), set(get_resource_cache_stats_provider().get_stats()))

def get_byte_length(value: Any) -> int:
    if False:
        print('Hello World!')
    'Return the byte length of the pickled value.'
    return asizeof(value)