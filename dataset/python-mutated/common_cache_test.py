"""Tests that are common to both st.cache_data and st.cache_resource"""
import threading
import time
import unittest
from datetime import timedelta
from typing import Any, List
from unittest.mock import MagicMock, Mock, patch
from parameterized import parameterized
import streamlit as st
from streamlit.runtime import Runtime
from streamlit.runtime.caching import CACHE_DATA_MESSAGE_REPLAY_CTX, CACHE_RESOURCE_MESSAGE_REPLAY_CTX, cache_data, cache_resource
from streamlit.runtime.caching.cache_errors import CacheReplayClosureError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cache_utils import CachedResult
from streamlit.runtime.caching.cached_message_replay import MultiCacheResults, _make_widget_key
from streamlit.runtime.caching.storage.dummy_cache_storage import MemoryCacheStorageManager
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.scriptrunner import ScriptRunContext, add_script_run_ctx, get_script_run_ctx, script_run_context
from streamlit.runtime.state import SafeSessionState, SessionState
from streamlit.testing.v1.app_test import AppTest
from tests.delta_generator_test_case import DeltaGeneratorTestCase
from tests.exception_capturing_thread import ExceptionCapturingThread, call_on_threads
from tests.streamlit.elements.image_test import create_image
from tests.testutil import create_mock_script_run_ctx

def get_text_or_block(delta):
    if False:
        return 10
    if delta.WhichOneof('type') == 'new_element':
        element = delta.new_element
        if element.WhichOneof('type') == 'text':
            return element.text.body
    elif delta.WhichOneof('type') == 'add_block':
        return 'new_block'

def as_cached_result(value: Any, cache_type: CacheType) -> MultiCacheResults:
    if False:
        print('Hello World!')
    'Creates cached results for a function that returned `value`\n    and did not execute any elements.\n    '
    result = CachedResult(value, [], st._main.id, st.sidebar.id)
    widget_key = _make_widget_key([], cache_type)
    d = {widget_key: result}
    initial = MultiCacheResults(set(), d)
    return initial

class CommonCacheTest(DeltaGeneratorTestCase):

    def tearDown(self):
        if False:
            while True:
                i = 10
        st.cache_data.clear()
        st.cache_resource.clear()
        ctx = script_run_context.get_script_run_ctx()
        if ctx is not None:
            ctx.widget_ids_this_run.clear()
            ctx.widget_user_keys_this_run.clear()
        super().tearDown()

    def get_text_delta_contents(self) -> List[str]:
        if False:
            while True:
                i = 10
        deltas = self.get_all_deltas_from_queue()
        text = [element.text.body for element in (delta.new_element for delta in deltas) if element.WhichOneof('type') == 'text']
        return text

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_simple(self, _, cache_decorator):
        if False:
            i = 10
            return i + 15

        @cache_decorator
        def foo():
            if False:
                print('Hello World!')
            return 42
        self.assertEqual(foo(), 42)
        self.assertEqual(foo(), 42)

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_multiple_int_like_floats(self, _, cache_decorator):
        if False:
            for i in range(10):
                print('nop')

        @cache_decorator
        def foo(x):
            if False:
                return 10
            return x
        self.assertEqual(foo(1.0), 1.0)
        self.assertEqual(foo(3.0), 3.0)

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_return_cached_object(self, _, cache_decorator):
        if False:
            return 10
        "If data has been cached, the cache function shouldn't be called."
        with patch.object(st, 'exception') as mock_exception:
            called = [False]

            @cache_decorator
            def f(x):
                if False:
                    for i in range(10):
                        print('nop')
                called[0] = True
                return x
            self.assertFalse(called[0])
            f(0)
            self.assertTrue(called[0])
            called = [False]
            f(0)
            self.assertFalse(called[0])
            f(1)
            self.assertTrue(called[0])
            mock_exception.assert_not_called()

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_mutate_args(self, _, cache_decorator):
        if False:
            i = 10
            return i + 15
        "Mutating an argument inside a cached function doesn't throw\n        an error (but it's probably not a great idea)."
        with patch.object(st, 'exception') as mock_exception:

            @cache_decorator
            def foo(d):
                if False:
                    i = 10
                    return i + 15
                d['answer'] += 1
                return d['answer']
            d = {'answer': 0}
            self.assertEqual(foo(d), 1)
            self.assertEqual(foo(d), 2)
            mock_exception.assert_not_called()

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_ignored_args(self, _, cache_decorator):
        if False:
            print('Hello World!')
        'Args prefixed with _ are not used as part of the cache key.'
        call_count = [0]

        @cache_decorator
        def foo(arg1, _arg2, *args, kwarg1, _kwarg2=None, **kwargs):
            if False:
                return 10
            call_count[0] += 1
        foo(1, 2, 3, kwarg1=4, _kwarg2=5, kwarg3=6, _kwarg4=7)
        self.assertEqual([1], call_count)
        foo(1, None, 3, kwarg1=4, _kwarg2=None, kwarg3=6, _kwarg4=None)
        self.assertEqual([1], call_count)
        foo(None, 2, 3, kwarg1=4, _kwarg2=5, kwarg3=6, _kwarg4=7)
        self.assertEqual([2], call_count)
        foo(1, 2, None, kwarg1=4, _kwarg2=5, kwarg3=6, _kwarg4=7)
        self.assertEqual([3], call_count)
        foo(1, 2, 3, kwarg1=None, _kwarg2=5, kwarg3=6, _kwarg4=7)
        self.assertEqual([4], call_count)
        foo(1, 2, 3, kwarg1=4, _kwarg2=5, kwarg3=None, _kwarg4=7)
        self.assertEqual([5], call_count)

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_cached_member_function(self, _, cache_decorator):
        if False:
            return 10
        'Our cache decorators can be applied to class member functions.'

        class TestClass:

            @cache_decorator
            def member_func(_self):
                if False:
                    while True:
                        i = 10
                return 'member func!'

            @classmethod
            @cache_decorator
            def class_method(cls):
                if False:
                    return 10
                return 'class method!'

            @staticmethod
            @cache_decorator
            def static_method():
                if False:
                    print('Hello World!')
                return 'static method!'
        obj = TestClass()
        self.assertEqual('member func!', obj.member_func())
        self.assertEqual('class method!', obj.class_method())
        self.assertEqual('static method!', obj.static_method())

    @parameterized.expand([('cache_data', cache_data, CACHE_DATA_MESSAGE_REPLAY_CTX), ('cache_resource', cache_resource, CACHE_RESOURCE_MESSAGE_REPLAY_CTX)])
    def test_cached_st_function_warning(self, _, cache_decorator, call_stack):
        if False:
            print('Hello World!')
        'Ensure we properly warn when interactive st.foo functions are called\n        inside a cached function.\n        '
        forward_msg_queue = ForwardMsgQueue()
        orig_report_ctx = get_script_run_ctx()
        add_script_run_ctx(threading.current_thread(), ScriptRunContext(session_id='test session id', _enqueue=forward_msg_queue.enqueue, query_string='', session_state=SafeSessionState(SessionState(), lambda : None), uploaded_file_mgr=MemoryUploadedFileManager('/mock/upload'), page_script_hash='', user_info={'email': 'test@test.com'}))
        with patch.object(call_stack, '_show_cached_st_function_warning') as warning:
            st.text('foo')
            warning.assert_not_called()

            @cache_decorator
            def cached_func():
                if False:
                    return 10
                st.text('Inside cached func')
            cached_func()
            warning.assert_not_called()
            warning.reset_mock()
            st.text('foo')
            warning.assert_not_called()

            @cache_decorator
            def outer():
                if False:
                    for i in range(10):
                        print('nop')

                @cache_decorator
                def inner():
                    if False:
                        print('Hello World!')
                    st.text('Inside nested cached func')
                return inner()
            outer()
            warning.assert_not_called()
            warning.reset_mock()
            with self.assertRaises(RuntimeError):

                @cache_decorator
                def cached_raise_error():
                    if False:
                        print('Hello World!')
                    st.text('About to throw')
                    raise RuntimeError('avast!')
                cached_raise_error()
            warning.assert_not_called()
            warning.reset_mock()
            st.text('foo')
            warning.assert_not_called()

            @cache_decorator
            def cached_widget():
                if False:
                    for i in range(10):
                        print('nop')
                st.button('Press me!')
            cached_widget()
            warning.assert_called()
            warning.reset_mock()
            st.text('foo')
            warning.assert_not_called()

            @cache_decorator(experimental_allow_widgets=True)
            def cached_widget_enabled():
                if False:
                    print('Hello World!')
                st.button('Press me too!')
            cached_widget_enabled()
            warning.assert_not_called()
            warning.reset_mock()
            st.text('foo')
            warning.assert_not_called()
            add_script_run_ctx(threading.current_thread(), orig_report_ctx)

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_cached_st_function_replay(self, _, cache_decorator):
        if False:
            return 10

        @cache_decorator
        def foo_replay(i):
            if False:
                return 10
            st.text(i)
            return i
        foo_replay(1)
        st.text('---')
        foo_replay(1)
        text = self.get_text_delta_contents()
        assert text == ['1', '---', '1']

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_cached_st_function_replay_nested(self, _, cache_decorator):
        if False:
            return 10

        @cache_decorator
        def inner(i):
            if False:
                print('Hello World!')
            st.text(i)

        @cache_decorator
        def outer(i):
            if False:
                while True:
                    i = 10
            inner(i)
            st.text(i + 10)
        outer(1)
        outer(1)
        st.text('---')
        inner(2)
        outer(2)
        st.text('---')
        outer(3)
        inner(3)
        text = self.get_text_delta_contents()
        assert text == ['1', '11', '1', '11', '---', '2', '2', '12', '---', '3', '13', '3']

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_cached_st_function_replay_outer_blocks(self, _, cache_decorator):
        if False:
            return 10

        @cache_decorator
        def foo(i):
            if False:
                i = 10
                return i + 15
            st.text(i)
            return i
        with st.container():
            foo(1)
            st.text('---')
            foo(1)
        text = self.get_text_delta_contents()
        assert text == ['1', '---', '1']

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_cached_st_function_replay_sidebar(self, _, cache_decorator):
        if False:
            print('Hello World!')

        @cache_decorator(show_spinner=False)
        def foo(i):
            if False:
                for i in range(10):
                    print('nop')
            st.sidebar.text(i)
            return i
        foo(1)
        st.text('---')
        foo(1)
        text = [get_text_or_block(delta) for delta in self.get_all_deltas_from_queue() if get_text_or_block(delta) is not None]
        assert text == ['1', '---', '1']
        paths = [msg.metadata.delta_path for msg in self.forward_msg_queue._queue if msg.HasField('delta')]
        assert paths == [[1, 0], [0, 0], [1, 1]]

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_cached_st_function_replay_inner_blocks(self, _, cache_decorator):
        if False:
            print('Hello World!')

        @cache_decorator(show_spinner=False)
        def foo(i):
            if False:
                while True:
                    i = 10
            with st.container():
                st.text(i)
                return i
        with st.container():
            st.text(0)
        st.text('---')
        with st.container():
            st.text(0)
        foo(1)
        st.text('---')
        foo(1)
        paths = [msg.metadata.delta_path for msg in self.forward_msg_queue._queue if msg.HasField('delta')]
        assert paths == [[0, 0], [0, 0, 0], [0, 1], [0, 2], [0, 2, 0], [0, 3], [0, 3, 0], [0, 4], [0, 5], [0, 5, 0]]

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_cached_st_function_replay_inner_direct(self, _, cache_decorator):
        if False:
            return 10

        @cache_decorator(show_spinner=False)
        def foo(i):
            if False:
                i = 10
                return i + 15
            cont = st.container()
            cont.text(i)
            return i
        foo(1)
        st.text('---')
        foo(1)
        text = self.get_text_delta_contents()
        assert text == ['1', '---', '1']
        paths = [msg.metadata.delta_path for msg in self.forward_msg_queue._queue if msg.HasField('delta')]
        assert paths == [[0, 0], [0, 0, 0], [0, 1], [0, 2], [0, 2, 0]]

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_cached_st_function_replay_outer_direct(self, _, cache_decorator):
        if False:
            print('Hello World!')
        cont = st.container()

        @cache_decorator
        def foo(i):
            if False:
                return 10
            cont.text(i)
            return i
        with self.assertRaises(CacheReplayClosureError):
            foo(1)
            st.text('---')
            foo(1)

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_cached_st_image_replay(self, _, cache_decorator):
        if False:
            for i in range(10):
                print('nop')
        'Basic sanity check that nothing blows up. This test assumes that\n        actual caching/replay functionality are covered by e2e tests that\n        can more easily test them.\n        '

        @cache_decorator
        def img_fn():
            if False:
                i = 10
                return i + 15
            st.image(create_image(10))
        img_fn()
        img_fn()

        @cache_decorator
        def img_fn_multi():
            if False:
                while True:
                    i = 10
            st.image([create_image(5), create_image(15), create_image(100)])
        img_fn_multi()
        img_fn_multi()

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_nested_widget_replay(self, _, cache_decorator):
        if False:
            return 10
        'Regression test for GH#5677'

        @cache_decorator(experimental_allow_widgets=True)
        def foo():
            if False:
                print('Hello World!')
            x = st.number_input('AAAA', 1, 100, 12)
            return x ** 2

        @cache_decorator(experimental_allow_widgets=True)
        def baz(y):
            if False:
                print('Hello World!')
            return foo() + y
        st.write(baz(10))

    @parameterized.expand([('cache_data', cache_data, cache_data.clear), ('cache_resource', cache_resource, cache_resource.clear)])
    def test_clear_all_caches(self, _, cache_decorator, clear_cache_func):
        if False:
            for i in range(10):
                print('nop')
        "Calling a cache's global `clear_all` function should remove all\n        items from all caches of the appropriate type.\n        "
        foo_vals = []

        @cache_decorator
        def foo(x):
            if False:
                while True:
                    i = 10
            foo_vals.append(x)
            return x
        bar_vals = []

        @cache_decorator
        def bar(x):
            if False:
                for i in range(10):
                    print('nop')
            bar_vals.append(x)
            return x
        (foo(0), foo(1), foo(2))
        (bar(0), bar(1), bar(2))
        self.assertEqual([0, 1, 2], foo_vals)
        self.assertEqual([0, 1, 2], bar_vals)
        clear_cache_func()
        (foo(0), foo(1), foo(2))
        (bar(0), bar(1), bar(2))
        self.assertEqual([0, 1, 2, 0, 1, 2], foo_vals)
        self.assertEqual([0, 1, 2, 0, 1, 2], bar_vals)

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_clear_single_cache(self, _, cache_decorator):
        if False:
            print('Hello World!')
        foo_call_count = [0]

        @cache_decorator
        def foo():
            if False:
                return 10
            foo_call_count[0] += 1
        bar_call_count = [0]

        @cache_decorator
        def bar():
            if False:
                return 10
            bar_call_count[0] += 1
        (foo(), foo(), foo())
        (bar(), bar(), bar())
        self.assertEqual(1, foo_call_count[0])
        self.assertEqual(1, bar_call_count[0])
        foo.clear()
        (foo(), foo(), foo())
        (bar(), bar(), bar())
        self.assertEqual(2, foo_call_count[0])
        self.assertEqual(1, bar_call_count[0])

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_without_spinner(self, _, cache_decorator):
        if False:
            print('Hello World!')
        'If the show_spinner flag is not set, the report queue should be\n        empty.\n        '

        @cache_decorator(show_spinner=False)
        def function_without_spinner(x: int) -> int:
            if False:
                i = 10
                return i + 15
            return x
        function_without_spinner(3)
        self.assertTrue(self.forward_msg_queue.is_empty())

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_with_spinner(self, _, cache_decorator):
        if False:
            for i in range(10):
                print('nop')
        'If the show_spinner flag is set, there should be one element in the\n        report queue.\n        '

        @cache_decorator(show_spinner=True)
        def function_with_spinner(x: int) -> int:
            if False:
                print('Hello World!')
            return x
        function_with_spinner(3)
        self.assertFalse(self.forward_msg_queue.is_empty())

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_with_custom_text_spinner(self, _, cache_decorator):
        if False:
            for i in range(10):
                print('nop')
        'If the show_spinner flag is set, there should be one element in the\n        report queue.\n        '

        @cache_decorator(show_spinner='CUSTOM_TEXT')
        def function_with_spinner_custom_text(x: int) -> int:
            if False:
                i = 10
                return i + 15
            return x
        function_with_spinner_custom_text(3)
        self.assertFalse(self.forward_msg_queue.is_empty())

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_with_empty_text_spinner(self, _, cache_decorator):
        if False:
            for i in range(10):
                print('nop')
        'If the show_spinner flag is set, even if it is empty text,\n        there should be one element in the report queue.\n        '

        @cache_decorator(show_spinner='')
        def function_with_spinner_empty_text(x: int) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return x
        function_with_spinner_empty_text(3)
        self.assertFalse(self.forward_msg_queue.is_empty())

class CommonCacheTTLTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        add_script_run_ctx(threading.current_thread(), create_mock_script_run_ctx())
        mock_runtime = MagicMock(spec=Runtime)
        mock_runtime.cache_storage_manager = MemoryCacheStorageManager()
        Runtime._instance = mock_runtime

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        cache_data.clear()
        cache_resource.clear()

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    @patch('streamlit.runtime.caching.cache_utils.TTLCACHE_TIMER')
    def test_ttl(self, _, cache_decorator, timer_patch: Mock):
        if False:
            return 10
        'Entries should expire after the given ttl.'
        one_day = 60 * 60 * 24
        foo_vals = []

        @cache_decorator(ttl=one_day)
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            foo_vals.append(x)
            return x
        bar_vals = []

        @cache_decorator(ttl=one_day * 2)
        def bar(x):
            if False:
                return 10
            bar_vals.append(x)
            return x
        timer_patch.return_value = 0
        foo(0)
        bar(0)
        self.assertEqual([0], foo_vals)
        self.assertEqual([0], bar_vals)
        timer_patch.return_value = one_day * 0.5
        foo(0)
        bar(0)
        self.assertEqual([0], foo_vals)
        self.assertEqual([0], bar_vals)
        timer_patch.return_value = one_day * 1.5
        foo(0)
        bar(0)
        self.assertEqual([0, 0], foo_vals)
        self.assertEqual([0], bar_vals)
        timer_patch.return_value = one_day * 2 + 1
        foo(0)
        bar(0)
        self.assertEqual([0, 0], foo_vals)
        self.assertEqual([0, 0], bar_vals)
        timer_patch.return_value = one_day * 2.5 + 1
        foo(0)
        bar(0)
        self.assertEqual([0, 0, 0], foo_vals)
        self.assertEqual([0, 0], bar_vals)

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    @patch('streamlit.runtime.caching.cache_utils.TTLCACHE_TIMER')
    def test_ttl_timedelta(self, _, cache_decorator, timer_patch: Mock):
        if False:
            return 10
        'Entries should expire after the given ttl.'
        one_day_seconds = 60 * 60 * 24
        one_day_timedelta = timedelta(days=1)
        two_days_timedelta = timedelta(days=2)
        foo_vals = []

        @cache_decorator(ttl=one_day_timedelta)
        def foo(x):
            if False:
                print('Hello World!')
            foo_vals.append(x)
            return x
        bar_vals = []

        @cache_decorator(ttl=two_days_timedelta)
        def bar(x):
            if False:
                while True:
                    i = 10
            bar_vals.append(x)
            return x
        timer_patch.return_value = 0
        foo(0)
        bar(0)
        self.assertEqual([0], foo_vals)
        self.assertEqual([0], bar_vals)
        timer_patch.return_value = one_day_seconds * 0.5
        foo(0)
        bar(0)
        self.assertEqual([0], foo_vals)
        self.assertEqual([0], bar_vals)
        timer_patch.return_value = one_day_seconds * 1.5
        foo(0)
        bar(0)
        self.assertEqual([0, 0], foo_vals)
        self.assertEqual([0], bar_vals)
        timer_patch.return_value = one_day_seconds * 2 + 1
        foo(0)
        bar(0)
        self.assertEqual([0, 0], foo_vals)
        self.assertEqual([0, 0], bar_vals)
        timer_patch.return_value = one_day_seconds * 2.5 + 1
        foo(0)
        bar(0)
        self.assertEqual([0, 0, 0], foo_vals)
        self.assertEqual([0, 0], bar_vals)

class CommonCacheThreadingTest(unittest.TestCase):
    NUM_THREADS = 50

    def setUp(self):
        if False:
            i = 10
            return i + 15
        mock_runtime = MagicMock(spec=Runtime)
        mock_runtime.cache_storage_manager = MemoryCacheStorageManager()
        Runtime._instance = mock_runtime

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        CACHE_DATA_MESSAGE_REPLAY_CTX._cached_func_stack = []
        CACHE_DATA_MESSAGE_REPLAY_CTX._suppress_st_function_warning = 0
        CACHE_RESOURCE_MESSAGE_REPLAY_CTX._cached_func_stack = []
        CACHE_RESOURCE_MESSAGE_REPLAY_CTX._suppress_st_function_warning = 0
        st.cache_data.clear()
        st.cache_resource.clear()
        ctx = script_run_context.get_script_run_ctx()
        if ctx is not None:
            ctx.widget_ids_this_run.clear()
            ctx.widget_user_keys_this_run.clear()
        super().tearDown()

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_get_cache(self, _, cache_decorator):
        if False:
            for i in range(10):
                print('nop')
        'Accessing a cached value is safe from multiple threads.'
        cached_func_call_count = [0]

        @cache_decorator
        def foo():
            if False:
                print('Hello World!')
            cached_func_call_count[0] += 1
            return 42

        def call_foo(_: int) -> None:
            if False:
                while True:
                    i = 10
            self.assertEqual(42, foo())
        call_on_threads(call_foo, self.NUM_THREADS)
        self.assertEqual(1, cached_func_call_count[0])

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_compute_value_only_once(self, _, cache_decorator):
        if False:
            return 10
        'Cached values should be computed only once, even if multiple sessions read from an\n        unwarmed cache simultaneously.\n        '
        cached_func_call_count = [0]

        @cache_decorator
        def foo():
            if False:
                while True:
                    i = 10
            self.assertEqual(0, cached_func_call_count[0], 'A cached value was computed multiple times!')
            cached_func_call_count[0] += 1
            time.sleep(0.25)
            return 42

        def call_foo(_: int) -> None:
            if False:
                return 10
            self.assertEqual(42, foo())
        call_on_threads(call_foo, num_threads=self.NUM_THREADS, timeout=0.5)

    @parameterized.expand([('cache_data', cache_data, cache_data.clear), ('cache_resource', cache_resource, cache_resource.clear)])
    def test_clear_all_caches(self, _, cache_decorator, clear_cache_func):
        if False:
            for i in range(10):
                print('nop')
        'Clearing all caches is safe to call from multiple threads.'

        @cache_decorator
        def foo():
            if False:
                print('Hello World!')
            return 42
        foo()

        def clear_caches(_: int) -> None:
            if False:
                return 10
            clear_cache_func()
        call_on_threads(clear_caches, self.NUM_THREADS)
        self.assertEqual(42, foo())

    @parameterized.expand([('cache_data', cache_data), ('cache_resource', cache_resource)])
    def test_clear_single_cache(self, _, cache_decorator):
        if False:
            return 10
        "It's safe to clear a single function cache from multiple threads."

        @cache_decorator
        def foo():
            if False:
                i = 10
                return i + 15
            return 42
        foo()

        def clear_foo(_: int) -> None:
            if False:
                for i in range(10):
                    print('nop')
            foo.clear()
        call_on_threads(clear_foo, self.NUM_THREADS)
        self.assertEqual(42, foo())

    @parameterized.expand([('cache_data', CACHE_DATA_MESSAGE_REPLAY_CTX), ('cache_resource', CACHE_RESOURCE_MESSAGE_REPLAY_CTX)])
    def test_multithreaded_call_stack(self, _, call_stack):
        if False:
            while True:
                i = 10
        'CachedFunctionCallStack works across multiple threads.'

        def get_counter():
            if False:
                print('Hello World!')
            return len(call_stack._cached_func_stack)

        def set_counter(val):
            if False:
                return 10
            call_stack._cached_func_stack = ['foo'] * val
        self.assertEqual(0, get_counter())
        set_counter(1)
        self.assertEqual(1, get_counter())
        values_in_thread = []

        def thread_test():
            if False:
                i = 10
                return i + 15
            values_in_thread.append(get_counter())
            set_counter(55)
            values_in_thread.append(get_counter())
        thread = ExceptionCapturingThread(target=thread_test)
        thread.start()
        thread.join()
        thread.assert_no_unhandled_exception()
        self.assertEqual([0, 55], values_in_thread)
        self.assertEqual(1, get_counter())

def test_dynamic_widget_replay():
    if False:
        i = 10
        return i + 15
    at = AppTest.from_file('test_data/cached_widget_replay_dynamic.py').run()
    assert at.checkbox.len == 1
    assert at.text[0].value == "['foo']"
    at.checkbox[0].check().run()
    assert at.multiselect.len == 1
    assert at.text[0].value == '[]'
    at.multiselect[0].select('baz').run()
    assert at.text[0].value == "['baz']"
    at.checkbox[0].uncheck().run()
    at.button[0].click().run()
    assert at.text[0].value == "['foo']"

def test_arrow_replay():
    if False:
        i = 10
        return i + 15
    'Regression test for https://github.com/streamlit/streamlit/issues/6103'
    at = AppTest.from_file('test_data/arrow_replay.py').run()
    assert not at.exception