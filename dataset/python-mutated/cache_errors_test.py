import threading
from parameterized import parameterized
import streamlit as st
from streamlit.elements import exception
from streamlit.proto.Exception_pb2 import Exception as ExceptionProto
from streamlit.runtime.caching.cache_errors import UnevaluatedDataFrameError, UnhashableParamError, UnserializableReturnValueError, get_return_value_type
from streamlit.runtime.caching.cache_utils import UNEVALUATED_DATAFRAME_TYPES
from tests import testutil
from tests.delta_generator_test_case import DeltaGeneratorTestCase
from tests.streamlit import pyspark_mocks, snowpark_mocks

class CacheErrorsTest(DeltaGeneratorTestCase):
    """Make sure user-visible error messages look correct.

    These errors are a little annoying to test, but they're important! So we
    are testing them word-for-word as much as possible. Even though this
    *feels* like an antipattern, it isn't: we're making sure the codepaths
    that pull useful debug info from the code are working.

    TODO: parameterize these tests for both memo + singleton
    """
    maxDiff = None

    def test_unhashable_type(self):
        if False:
            return 10

        @st.cache_data
        def unhashable_type_func(lock: threading.Lock):
            if False:
                for i in range(10):
                    print('nop')
            return str(lock)
        with self.assertRaises(UnhashableParamError) as cm:
            unhashable_type_func(threading.Lock())
        ep = ExceptionProto()
        exception.marshall(ep, cm.exception)
        self.assertEqual(ep.type, 'UnhashableParamError')
        expected_message = "\nCannot hash argument 'lock' (of type `_thread.lock`) in 'unhashable_type_func'.\n\nTo address this, you can tell Streamlit not to hash this argument by adding a\nleading underscore to the argument's name in the function signature:\n\n```\n@st.cache_data\ndef unhashable_type_func(_lock, ...):\n    ...\n```\n                    "
        self.assertEqual(testutil.normalize_md(expected_message), testutil.normalize_md(ep.message))
        self.assertEqual(ep.message_is_markdown, True)
        self.assertEqual(ep.is_warning, False)

    def test_unserializable_return_value_error(self):
        if False:
            while True:
                i = 10

        @st.cache_data
        def unserializable_return_value_func():
            if False:
                for i in range(10):
                    print('nop')
            return threading.Lock()
        with self.assertRaises(UnserializableReturnValueError) as cm:
            unserializable_return_value_func()
        ep = ExceptionProto()
        exception.marshall(ep, cm.exception)
        self.assertEqual(ep.type, 'UnserializableReturnValueError')
        expected_message = f'\n            Cannot serialize the return value (of type {get_return_value_type(return_value=threading.Lock())}) in `unserializable_return_value_func()`.\n            `st.cache_data` uses [pickle](https://docs.python.org/3/library/pickle.html) to\n            serialize the function’s return value and safely store it in the cache without mutating the original object. Please convert the return value to a pickle-serializable type.\n            If you want to cache unserializable objects such as database connections or Tensorflow\n            sessions, use `st.cache_resource` instead (see [our docs](https://docs.streamlit.io/library/advanced-features/caching) for differences).'
        self.assertEqual(testutil.normalize_md(expected_message), testutil.normalize_md(ep.message))
        self.assertEqual(ep.message_is_markdown, True)
        self.assertEqual(ep.is_warning, False)

    @parameterized.expand(UNEVALUATED_DATAFRAME_TYPES)
    def test_unevaluated_dataframe_error(self, type_name):
        if False:
            for i in range(10):
                print('nop')
        if 'snowpark.table.Table' in type_name:
            to_return = snowpark_mocks.Table()
        elif 'snowpark.dataframe.DataFrame' in type_name:
            to_return = snowpark_mocks.DataFrame()
        else:
            to_return = pyspark_mocks.create_pyspark_dataframe_with_mocked_personal_data()

        @st.experimental_memo
        def unevaluated_dataframe_func():
            if False:
                return 10
            return to_return
        with self.assertRaises(UnevaluatedDataFrameError) as cm:
            unevaluated_dataframe_func()
        ep = ExceptionProto()
        exception.marshall(ep, cm.exception)
        self.assertEqual(ep.type, 'UnevaluatedDataFrameError')
        expected_message = 'Please call `collect()` or `to_pandas()` on the dataframe before returning it'
        self.assertTrue(expected_message in ep.message)
        self.assertEqual(ep.message_is_markdown, True)
        self.assertEqual(ep.is_warning, False)