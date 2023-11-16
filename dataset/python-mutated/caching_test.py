"""st.caching unit tests."""
import threading
import types
from unittest.mock import Mock, patch
from parameterized import parameterized
import streamlit as st
from streamlit.elements import exception
from streamlit.proto.Exception_pb2 import Exception as ExceptionProto
from streamlit.runtime.legacy_caching import caching, hashing
from tests import testutil
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class NotHashable:
    """A class that is not hashable."""

    def __reduce__(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

class CacheTest(DeltaGeneratorTestCase):

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        caching._cache_info.cached_func_stack = []
        caching._cache_info.suppress_st_function_warning = 0
        super().tearDown()

    def test_simple(self):
        if False:
            print('Hello World!')

        @st.cache
        def foo():
            if False:
                while True:
                    i = 10
            return 42
        self.assertEqual(foo(), 42)
        self.assertEqual(foo(), 42)

    def test_multiple_int_like_floats(self):
        if False:
            return 10

        @st.cache
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return x
        self.assertEqual(foo(1.0), 1.0)
        self.assertEqual(foo(3.0), 3.0)

    @patch.object(st, 'exception')
    def test_args(self, exception):
        if False:
            i = 10
            return i + 15
        called = [False]

        @st.cache
        def f(x):
            if False:
                while True:
                    i = 10
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
        exception.assert_not_called()

    @patch.object(st, 'exception')
    def test_mutate_return(self, exception):
        if False:
            return 10

        @st.cache
        def f():
            if False:
                i = 10
                return i + 15
            return [0, 1]
        r = f()
        r[0] = 1
        exception.assert_not_called()
        r2 = f()
        exception.assert_called()
        self.assertEqual(r, r2)

    @patch.object(st, 'exception')
    def test_mutate_args(self, exception):
        if False:
            while True:
                i = 10

        @st.cache
        def foo(d):
            if False:
                return 10
            d['answer'] += 1
            return d['answer']
        d = {'answer': 0}
        self.assertNotEqual(foo(d), foo(d))
        exception.assert_not_called()

    @patch('streamlit.runtime.legacy_caching.caching.show_deprecation_warning', Mock())
    @patch('streamlit.runtime.legacy_caching.caching._show_cached_st_function_warning')
    def test_cached_st_function_warning(self, warning):
        if False:
            return 10
        st.text('foo')
        warning.assert_not_called()

        @st.cache
        def cached_func():
            if False:
                print('Hello World!')
            st.text('Inside cached func')
        cached_func()
        warning.assert_called_once()
        warning.reset_mock()
        st.text('foo')
        warning.assert_not_called()

        @st.cache(suppress_st_warning=True)
        def suppressed_cached_func():
            if False:
                i = 10
                return i + 15
            st.text('No warnings here!')
        suppressed_cached_func()
        warning.assert_not_called()

        @st.cache
        def outer():
            if False:
                print('Hello World!')

            @st.cache
            def inner():
                if False:
                    i = 10
                    return i + 15
                st.text('Inside nested cached func')
            return inner()
        outer()
        warning.assert_called_once()
        warning.reset_mock()
        with self.assertRaises(RuntimeError):

            @st.cache
            def cached_raise_error():
                if False:
                    return 10
                st.text('About to throw')
                raise RuntimeError('avast!')
            cached_raise_error()
        warning.assert_called_once()
        warning.reset_mock()
        st.text('foo')
        warning.assert_not_called()

        @st.cache
        def cached_widget():
            if False:
                for i in range(10):
                    print('nop')
            st.button('Press me!')
        cached_widget()
        warning.assert_called_once()
        warning.reset_mock()
        st.text('foo')
        warning.assert_not_called()

    def test_multithread_stack(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that cached_func_stack behaves properly in multiple threads.'

        def get_counter():
            if False:
                i = 10
                return i + 15
            return len(caching._cache_info.cached_func_stack)

        def set_counter(val):
            if False:
                i = 10
                return i + 15
            caching._cache_info.cached_func_stack = ['foo'] * val
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
        thread = threading.Thread(target=thread_test)
        thread.start()
        thread.join()
        self.assertEqual([0, 55], values_in_thread)
        self.assertEqual(1, get_counter())

    def test_max_size(self):
        if False:
            for i in range(10):
                print('nop')
        'The oldest object should be evicted when maxsize is reached.'
        foo_vals = []

        @st.cache(max_entries=2)
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            foo_vals.append(x)
            return x
        bar_vals = []

        @st.cache(max_entries=3)
        def bar(x):
            if False:
                while True:
                    i = 10
            bar_vals.append(x)
            return x
        self.assertEqual([], foo_vals)
        self.assertEqual([], bar_vals)
        (foo(0), foo(1))
        (bar(0), bar(1))
        self.assertEqual([0, 1], foo_vals)
        self.assertEqual([0, 1], bar_vals)
        (foo(0), foo(1))
        (bar(0), bar(1))
        self.assertEqual([0, 1], foo_vals)
        self.assertEqual([0, 1], bar_vals)
        foo(2)
        bar(2)
        (foo(1), foo(0))
        (bar(1), bar(0))
        self.assertEqual([0, 1, 2, 0], foo_vals)
        self.assertEqual([0, 1, 2], bar_vals)

    def test_no_max_size(self):
        if False:
            while True:
                i = 10
        'If max_size is None, the cache is unbounded.'
        called_values = []

        @st.cache(max_entries=None)
        def f(x):
            if False:
                i = 10
                return i + 15
            called_values.append(x)
            return x
        for ii in range(256):
            f(ii)
        called_values = []
        for ii in range(256):
            f(ii)
        self.assertEqual([], called_values)

    @patch('streamlit.runtime.legacy_caching.caching._TTLCACHE_TIMER')
    def test_ttl(self, timer_patch):
        if False:
            i = 10
            return i + 15
        'Entries should expire after the given ttl.'
        foo_vals = []

        @st.cache(ttl=1)
        def foo(x):
            if False:
                return 10
            foo_vals.append(x)
            return x
        bar_vals = []

        @st.cache(ttl=5)
        def bar(x):
            if False:
                for i in range(10):
                    print('nop')
            bar_vals.append(x)
            return x
        timer_patch.return_value = 0
        foo(0)
        bar(0)
        self.assertEqual([0], foo_vals)
        self.assertEqual([0], bar_vals)
        timer_patch.return_value = 0.5
        foo(0)
        bar(0)
        self.assertEqual([0], foo_vals)
        self.assertEqual([0], bar_vals)
        timer_patch.return_value = 1.5
        foo(0)
        bar(0)
        self.assertEqual([0, 0], foo_vals)
        self.assertEqual([0], bar_vals)

    def test_clear_cache(self):
        if False:
            for i in range(10):
                print('nop')
        'Clear cache should do its thing.'
        foo_vals = []

        @st.cache
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            foo_vals.append(x)
            return x
        bar_vals = []

        @st.cache
        def bar(x):
            if False:
                i = 10
                return i + 15
            bar_vals.append(x)
            return x
        (foo(0), foo(1), foo(2))
        (bar(0), bar(1), bar(2))
        self.assertEqual([0, 1, 2], foo_vals)
        self.assertEqual([0, 1, 2], bar_vals)
        caching.clear_cache()
        (foo(0), foo(1), foo(2))
        (bar(0), bar(1), bar(2))
        self.assertEqual([0, 1, 2, 0, 1, 2], foo_vals)
        self.assertEqual([0, 1, 2, 0, 1, 2], bar_vals)

    def test_unique_function_caches(self):
        if False:
            print('Hello World!')
        'Each function should have its own cache, even if it has an\n        identical body and arguments to another cached function.\n        '

        @st.cache
        def foo():
            if False:
                while True:
                    i = 10
            return []

        @st.cache
        def bar():
            if False:
                print('Hello World!')
            return []
        id_foo = id(foo())
        id_bar = id(bar())
        self.assertNotEqual(id_foo, id_bar)

    def test_function_body_uses_hashfuncs(self):
        if False:
            print('Hello World!')
        hash_func = Mock(return_value=None)
        dict_gen = {1: (x for x in range(1))}

        @st.cache(hash_funcs={'builtins.generator': hash_func})
        def foo(arg):
            if False:
                return 10
            print(dict_gen)
            return []
        foo(1)
        foo(2)
        hash_func.assert_called_once()

    def test_function_body_uses_nested_listcomps(self):
        if False:
            print('Hello World!')

        @st.cache()
        def foo(arg):
            if False:
                for i in range(10):
                    print('nop')
            production = [[outer + inner for inner in range(3)] for outer in range(3)]
            return production
        self.assertEqual(foo(1), [[0, 1, 2], [1, 2, 3], [2, 3, 4]])

    def test_function_name_does_not_use_hashfuncs(self):
        if False:
            i = 10
            return i + 15
        "Hash funcs should only be used on arguments to a function,\n        and not when computing the key for a function's unique MemCache.\n        "
        str_hash_func = Mock(return_value=None)

        @st.cache(hash_funcs={str: str_hash_func})
        def foo(string_arg):
            if False:
                for i in range(10):
                    print('nop')
            return []
        foo('ahoy')
        str_hash_func.assert_called_once_with('ahoy')

class CacheErrorsTest(DeltaGeneratorTestCase):
    """Make sure user-visible error messages look correct.

    These errors are a little annoying to test, but they're important! So we
    are testing them word-for-word as much as possible. Even though this
    *feels* like an antipattern, it isn't: we're making sure the codepaths
    that pull useful debug info from the code are working.
    """

    @patch('streamlit.runtime.legacy_caching.caching.show_deprecation_warning', Mock())
    def test_st_warning_text(self):
        if False:
            while True:
                i = 10

        @st.cache
        def st_warning_text_func():
            if False:
                while True:
                    i = 10
            st.markdown('hi')
        st_warning_text_func()
        el = self.get_delta_from_queue(-2).new_element
        self.assertEqual(el.exception.type, 'CachedStFunctionWarning')
        self.assertEqual(normalize_md(el.exception.message), normalize_md('\nYour script uses `st.markdown()` or `st.write()` to write to your Streamlit app\nfrom within some cached code at `st_warning_text_func()`. This code will only be\ncalled when we detect a cache "miss", which can lead to unexpected results.\n\nHow to fix this:\n* Move the `st.markdown()` or `st.write()` call outside `st_warning_text_func()`.\n* Or, if you know what you\'re doing, use `@st.cache(suppress_st_warning=True)`\nto suppress the warning.\n        '))
        self.assertNotEqual(len(el.exception.stack_trace), 0)
        self.assertEqual(el.exception.message_is_markdown, True)
        self.assertEqual(el.exception.is_warning, True)
        el = self.get_delta_from_queue(-1).new_element
        self.assertEqual(el.markdown.body, 'hi')

    @parameterized.expand([(True,), (False,)])
    @patch('streamlit.runtime.legacy_caching.caching.show_deprecation_warning', Mock())
    def test_mutation_warning_text(self, show_error_details: bool):
        if False:
            return 10
        with testutil.patch_config_options({'client.showErrorDetails': show_error_details}):

            @st.cache
            def mutation_warning_func():
                if False:
                    print('Hello World!')
                return []
            a = mutation_warning_func()
            a.append('mutated!')
            mutation_warning_func()
            if show_error_details:
                el = self.get_delta_from_queue(-1).new_element
                self.assertEqual(el.exception.type, 'CachedObjectMutationWarning')
                self.assertEqual(normalize_md(el.exception.message), normalize_md("\nReturn value of `mutation_warning_func()` was mutated between runs.\n\nBy default, Streamlit's cache should be treated as immutable, or it may behave\nin unexpected ways. You received this warning because Streamlit detected that\nan object returned by `mutation_warning_func()` was mutated outside of\n`mutation_warning_func()`.\n\nHow to fix this:\n* If you did not mean to mutate that return value:\n  - If possible, inspect your code to find and remove that mutation.\n  - Otherwise, you could also clone the returned value so you can freely\n    mutate it.\n* If you actually meant to mutate the return value and know the consequences of\ndoing so, annotate the function with `@st.cache(allow_output_mutation=True)`.\n\nFor more information and detailed solutions check out [our\ndocumentation.](https://docs.streamlit.io/library/advanced-features/caching)\n                    "))
                self.assertNotEqual(len(el.exception.stack_trace), 0)
                self.assertEqual(el.exception.message_is_markdown, True)
                self.assertEqual(el.exception.is_warning, True)
            else:
                el = self.get_delta_from_queue(-1).new_element
                self.assertEqual(el.WhichOneof('type'), 'exception')

    def test_unhashable_type(self):
        if False:
            i = 10
            return i + 15

        @st.cache
        def unhashable_type_func():
            if False:
                for i in range(10):
                    print('nop')
            return NotHashable()
        with self.assertRaises(hashing.UnhashableTypeError) as cm:
            unhashable_type_func()
        ep = ExceptionProto()
        exception.marshall(ep, cm.exception)
        self.assertEqual(ep.type, 'UnhashableTypeError')
        self.assertTrue(normalize_md(ep.message).startswith(normalize_md("\nCannot hash object of type `tests.streamlit.runtime.legacy_caching.caching_test.NotHashable`, found in the return value of\n`unhashable_type_func()`.\n\nWhile caching the return value of `unhashable_type_func()`, Streamlit encountered an\nobject of type `tests.streamlit.runtime.legacy_caching.caching_test.NotHashable`, which it does not know how to hash.\n\nTo address this, please try helping Streamlit understand how to hash that type\nby passing the `hash_funcs` argument into `@st.cache`. For example:\n\n```\n@st.cache(hash_funcs={tests.streamlit.runtime.legacy_caching.caching_test.NotHashable: my_hash_func})\ndef my_func(...):\n    ...\n```\n\nIf you don't know where the object of type `tests.streamlit.runtime.legacy_caching.caching_test.NotHashable` is coming\nfrom, try looking at the hash chain below for an object that you do recognize,\nthen pass that to `hash_funcs` instead:\n\n```\nObject of type tests.streamlit.runtime.legacy_caching.caching_test.NotHashable:\n                    ")))
        self.assertEqual(ep.message_is_markdown, True)
        self.assertEqual(ep.is_warning, False)

    def test_hash_funcs_acceptable_keys(self):
        if False:
            print('Hello World!')

        @st.cache
        def unhashable_type_func():
            if False:
                print('Hello World!')
            return (x for x in range(1))

        @st.cache(hash_funcs={types.GeneratorType: id})
        def hf_key_as_type():
            if False:
                for i in range(10):
                    print('nop')
            return (x for x in range(1))

        @st.cache(hash_funcs={'builtins.generator': id})
        def hf_key_as_str():
            if False:
                print('Hello World!')
            return (x for x in range(1))
        with self.assertRaises(hashing.UnhashableTypeError) as cm:
            unhashable_type_func()
        self.assertEqual(list(hf_key_as_type()), list(hf_key_as_str()))

    def test_user_hash_error(self):
        if False:
            i = 10
            return i + 15

        class MyObj(object):
            pass

        def bad_hash_func(x):
            if False:
                while True:
                    i = 10
            x += 10
            return x

        @st.cache(hash_funcs={MyObj: bad_hash_func})
        def user_hash_error_func(x):
            if False:
                for i in range(10):
                    print('nop')
            pass
        with self.assertRaises(hashing.UserHashError) as cm:
            my_obj = MyObj()
            user_hash_error_func(my_obj)
        ep = ExceptionProto()
        exception.marshall(ep, cm.exception)
        self.assertEqual(ep.type, 'TypeError')
        self.assertTrue(normalize_md(ep.message).startswith(normalize_md("\nunsupported operand type(s) for +=: 'MyObj' and 'int'\n\nThis error is likely due to a bug in `bad_hash_func()`, which is a user-defined\nhash function that was passed into the `@st.cache` decorator of `user_hash_error_func()`.\n\n`bad_hash_func()` failed when hashing an object of type\n`tests.streamlit.runtime.legacy_caching.caching_test.CacheErrorsTest.test_user_hash_error.<locals>.MyObj`.  If you\ndon't know where that object is coming from, try looking at the hash chain below\nfor an object that you do recognize, then pass that to `hash_funcs` instead:\n\n```\nObject of type tests.streamlit.runtime.legacy_caching.caching_test.CacheErrorsTest.test_user_hash_error.<locals>.MyObj:\n<tests.streamlit.runtime.legacy_caching.caching_test.CacheErrorsTest.test_user_hash_error.<locals>.MyObj object at\n                    ")))
        self.assertEqual(ep.message_is_markdown, True)
        self.assertEqual(ep.is_warning, False)

    @patch('streamlit.runtime.legacy_caching.caching.show_deprecation_warning')
    def test_type_specific_deprecation_warning(self, show_deprecation_warning: Mock) -> None:
        if False:
            while True:
                i = 10
        'Calling a @st.cache function shows a type-specific deprecation warning for certain types.'

        @st.cache
        def func():
            if False:
                for i in range(10):
                    print('nop')
            return 42
        show_deprecation_warning.assert_not_called()
        self.assertEqual(42, func())
        expected_message = "`st.cache` is deprecated. Please use one of Streamlit's new caching commands,\n`st.cache_data` or `st.cache_resource`. Based on this function's return value\nof type `int`, we recommend using `st.cache_data`.\n\nMore information [in our docs](https://docs.streamlit.io/library/advanced-features/caching)."
        show_deprecation_warning.assert_called_once_with(expected_message)

    @patch('streamlit.runtime.legacy_caching.caching.show_deprecation_warning')
    def test_generic_deprecation_warning(self, show_deprecation_warning: Mock) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Calling a @st.cache function shows a generic deprecation warning for other types.'

        class MockClass:
            pass

        @st.cache
        def func():
            if False:
                print('Hello World!')
            return MockClass()
        show_deprecation_warning.assert_not_called()
        self.assertIsInstance(func(), MockClass)
        expected_message = "`st.cache` is deprecated. Please use one of Streamlit's new caching commands,\n`st.cache_data` or `st.cache_resource`.\n\nMore information [in our docs](https://docs.streamlit.io/library/advanced-features/caching)."
        show_deprecation_warning.assert_called_once_with(expected_message)

def normalize_md(txt):
    if False:
        i = 10
        return i + 15
    'Replace newlines *inside paragraphs* with spaces.\n\n    Consecutive lines of text are considered part of the same paragraph\n    in Markdown. So this function joins those into a single line to make the\n    test robust to changes in text wrapping.\n\n    NOTE: This function doesn\'t attempt to be 100% grammatically correct\n    Markdown! It\'s just supposed to be "correct enough" for tests to pass. For\n    example, when we guard "\n\n" from being converted, we really should be\n    guarding for RegEx("\n\n+") instead. But that doesn\'t matter for our tests.\n    '
    txt = txt.replace('\n\n', 'OMG_NEWLINE')
    txt = txt.replace('\n*', 'OMG_STAR')
    txt = txt.replace('\n-', 'OMG_HYPHEN')
    txt = txt.replace(']\n(', 'OMG_LINK')
    txt = txt.replace('\n', ' ')
    txt = txt.replace('OMG_NEWLINE', '\n\n')
    txt = txt.replace('OMG_STAR', '\n*')
    txt = txt.replace('OMG_HYPHEN', '\n-')
    txt = txt.replace('OMG_LINK', '](')
    return txt.strip()

def test_cache_stats_provider():
    if False:
        while True:
            i = 10
    caches = caching._mem_caches
    caches.clear()
    init_size = sum((stat.byte_length for stat in caches.get_stats()))
    assert init_size == 0
    assert len(caches.get_stats()) == 0

    @st.cache
    def foo():
        if False:
            return 10
        return 42
    foo()
    new_size = sum((stat.byte_length for stat in caches.get_stats()))
    assert new_size > 0
    assert len(caches.get_stats()) == 1
    foo()
    new_size_2 = sum((stat.byte_length for stat in caches.get_stats()))
    assert new_size_2 == new_size

    @st.cache
    def bar(i):
        if False:
            for i in range(10):
                print('nop')
        return 0
    bar(0)
    new_size_3 = sum((stat.byte_length for stat in caches.get_stats()))
    assert new_size_3 > new_size_2
    assert len(caches.get_stats()) == 2
    bar(1)
    new_size_4 = sum((stat.byte_length for stat in caches.get_stats()))
    assert new_size_4 > new_size_3
    assert len(caches.get_stats()) == 3