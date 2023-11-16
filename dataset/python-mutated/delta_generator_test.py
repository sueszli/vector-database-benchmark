"""DeltaGenerator Unittest."""
import functools
import inspect
import json
import logging
import re
import unittest
from unittest.mock import MagicMock, patch
import pytest
from parameterized import parameterized
import streamlit as st
import streamlit.delta_generator as delta_generator
import streamlit.runtime.state.widgets as w
from streamlit.cursor import LockedCursor, make_delta_path
from streamlit.delta_generator import DeltaGenerator
from streamlit.errors import DuplicateWidgetID, StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto.Element_pb2 import Element
from streamlit.proto.Empty_pb2 import Empty as EmptyProto
from streamlit.proto.RootContainer_pb2 import RootContainer
from streamlit.proto.Text_pb2 import Text as TextProto
from streamlit.proto.TextArea_pb2 import TextArea
from streamlit.proto.TextInput_pb2 import TextInput
from streamlit.runtime.state.common import compute_widget_id
from streamlit.runtime.state.widgets import _build_duplicate_widget_message
from tests.delta_generator_test_case import DeltaGeneratorTestCase

def identity(x):
    if False:
        i = 10
        return i + 15
    return x
register_widget = functools.partial(w.register_widget, deserializer=lambda x, s: x, serializer=identity)

class RunWarningTest(unittest.TestCase):

    @patch('streamlit.runtime.Runtime.exists', MagicMock(return_value=False))
    def test_run_warning_presence(self):
        if False:
            for i in range(10):
                print('nop')
        'Using Streamlit without `streamlit run` produces a warning.'
        with self.assertLogs('streamlit', level=logging.WARNING) as logs:
            delta_generator._use_warning_has_been_displayed = False
            st.write('Using delta generator')
            output = ''.join(logs.output)
            self.assertEqual(len(re.findall('streamlit run', output)), 1)

    @patch('streamlit.runtime.Runtime.exists', MagicMock(return_value=True))
    def test_run_warning_absence(self):
        if False:
            while True:
                i = 10
        'Using Streamlit through the CLI results in a Runtime being instantiated,\n        so it produces no usage warning.'
        with self.assertLogs('streamlit', level=logging.WARNING) as logs:
            delta_generator._use_warning_has_been_displayed = False
            st.write('Using delta generator')
            get_logger('root').warning('irrelevant warning so assertLogs passes')
            self.assertNotRegex(''.join(logs.output), 'streamlit run')

    def test_public_api(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that we don't accidentally remove (or add) symbols\n        to the public `DeltaGenerator` API.\n        "
        api = {name for (name, _) in inspect.getmembers(DeltaGenerator) if not name.startswith('_')}
        self.assertEqual(api, {'add_rows', 'altair_chart', 'area_chart', 'audio', 'balloons', 'bar_chart', 'bokeh_chart', 'button', 'camera_input', 'caption', 'chat_input', 'chat_message', 'checkbox', 'code', 'color_picker', 'columns', 'container', 'dataframe', 'data_editor', 'date_input', 'dg', 'divider', 'download_button', 'empty', 'error', 'exception', 'expander', 'experimental_data_editor', 'file_uploader', 'form', 'form_submit_button', 'graphviz_chart', 'header', 'help', 'id', 'image', 'info', 'json', 'latex', 'line_chart', 'link_button', 'map', 'markdown', 'metric', 'multiselect', 'number_input', 'plotly_chart', 'progress', 'pydeck_chart', 'pyplot', 'radio', 'scatter_chart', 'select_slider', 'selectbox', 'slider', 'snow', 'subheader', 'success', 'status', 'table', 'tabs', 'text', 'text_area', 'text_input', 'time_input', 'title', 'toast', 'toggle', 'vega_lite_chart', 'video', 'warning', 'write'})

class DeltaGeneratorTest(DeltaGeneratorTestCase):
    """Test streamlit.delta_generator methods."""

    def test_nonexistent_method(self):
        if False:
            return 10
        with self.assertRaises(Exception) as ctx:
            st.sidebar.non_existing()
        self.assertEqual(str(ctx.exception), '`non_existing()` is not a valid Streamlit command.')

    def test_sidebar_nonexistent_method(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(Exception) as ctx:
            st.sidebar.echo()
        self.assertEqual(str(ctx.exception), 'Method `echo()` does not exist for `st.sidebar`. Did you mean `st.echo()`?')

    def set_widget_requires_args(self):
        if False:
            i = 10
            return i + 15
        st.text_input()
        c = self.get_delta_from_queue().new_element.exception
        self.assertEqual(c.type, 'TypeError')

    def test_duplicate_widget_id_error(self):
        if False:
            print('Hello World!')
        'Multiple widgets with the same generated key should report an error.'
        widgets = {'button': lambda key=None: st.button('', key=key), 'checkbox': lambda key=None: st.checkbox('', key=key), 'multiselect': lambda key=None: st.multiselect('', options=[1, 2], key=key), 'radio': lambda key=None: st.radio('', options=[1, 2], key=key), 'selectbox': lambda key=None: st.selectbox('', options=[1, 2], key=key), 'slider': lambda key=None: st.slider('', key=key), 'text_area': lambda key=None: st.text_area('', key=key), 'text_input': lambda key=None: st.text_input('', key=key), 'time_input': lambda key=None: st.time_input('', key=key), 'date_input': lambda key=None: st.date_input('', key=key), 'number_input': lambda key=None: st.number_input('', key=key)}
        for (widget_type, create_widget) in widgets.items():
            create_widget()
            with self.assertRaises(DuplicateWidgetID) as ctx:
                create_widget()
            self.assertEqual(_build_duplicate_widget_message(widget_func_name=widget_type, user_key=None), str(ctx.exception))
        for (widget_type, create_widget) in widgets.items():
            create_widget(widget_type)
            with self.assertRaises(DuplicateWidgetID) as ctx:
                create_widget(widget_type)
            self.assertEqual(_build_duplicate_widget_message(widget_func_name=widget_type, user_key=widget_type), str(ctx.exception))

    def test_duplicate_widget_id_error_when_user_key_specified(self):
        if False:
            for i in range(10):
                print('nop')
        'Multiple widgets with the different generated key, but same user specified\n        key should report an error.\n        '
        widgets = {'button': lambda key=None, label='': st.button(label=label, key=key), 'checkbox': lambda key=None, label='': st.checkbox(label=label, key=key), 'multiselect': lambda key=None, label='': st.multiselect(label=label, options=[1, 2], key=key), 'radio': lambda key=None, label='': st.radio(label=label, options=[1, 2], key=key), 'selectbox': lambda key=None, label='': st.selectbox(label=label, options=[1, 2], key=key), 'slider': lambda key=None, label='': st.slider(label=label, key=key), 'text_area': lambda key=None, label='': st.text_area(label=label, key=key), 'text_input': lambda key=None, label='': st.text_input(label=label, key=key), 'time_input': lambda key=None, label='': st.time_input(label=label, key=key), 'date_input': lambda key=None, label='': st.date_input(label=label, key=key), 'number_input': lambda key=None, label='': st.number_input(label=label, key=key)}
        for (widget_type, create_widget) in widgets.items():
            user_key = widget_type
            create_widget(label='LABEL_A', key=user_key)
            with self.assertRaises(DuplicateWidgetID) as ctx:
                create_widget(label='LABEL_B', key=user_key)
            self.assertEqual(_build_duplicate_widget_message(widget_func_name=widget_type, user_key=user_key), str(ctx.exception))

class DeltaGeneratorClassTest(DeltaGeneratorTestCase):
    """Test DeltaGenerator Class."""

    def test_constructor(self):
        if False:
            i = 10
            return i + 15
        'Test default DeltaGenerator().'
        dg = DeltaGenerator()
        self.assertFalse(dg._cursor.is_locked)
        self.assertEqual(dg._cursor.index, 0)

    def test_constructor_with_id(self):
        if False:
            i = 10
            return i + 15
        'Test DeltaGenerator() with an id.'
        cursor = LockedCursor(root_container=RootContainer.MAIN, index=1234)
        dg = DeltaGenerator(root_container=RootContainer.MAIN, cursor=cursor)
        self.assertTrue(dg._cursor.is_locked)
        self.assertEqual(dg._cursor.index, 1234)

    def test_enqueue_null(self):
        if False:
            return 10
        dg = DeltaGenerator(root_container=None)
        new_dg = dg._enqueue('empty', EmptyProto())
        self.assertEqual(dg, new_dg)

    @parameterized.expand([(RootContainer.MAIN,), (RootContainer.SIDEBAR,)])
    def test_enqueue(self, container):
        if False:
            for i in range(10):
                print('nop')
        dg = DeltaGenerator(root_container=container)
        self.assertEqual(0, dg._cursor.index)
        self.assertEqual(container, dg._root_container)
        test_data = 'some test data'
        text_proto = TextProto()
        text_proto.body = test_data
        new_dg = dg._enqueue('text', text_proto)
        self.assertNotEqual(dg, new_dg)
        self.assertEqual(1, dg._cursor.index)
        self.assertEqual(container, new_dg._root_container)
        element = self.get_delta_from_queue().new_element
        self.assertEqual(element.text.body, test_data)

    def test_enqueue_same_id(self):
        if False:
            while True:
                i = 10
        cursor = LockedCursor(root_container=RootContainer.MAIN, index=123)
        dg = DeltaGenerator(root_container=RootContainer.MAIN, cursor=cursor)
        self.assertEqual(123, dg._cursor.index)
        test_data = 'some test data'
        text_proto = TextProto()
        text_proto.body = test_data
        new_dg = dg._enqueue('text', text_proto)
        self.assertEqual(dg._cursor, new_dg._cursor)
        msg = self.get_message_from_queue()
        self.assertEqual(make_delta_path(RootContainer.MAIN, (), 123), msg.metadata.delta_path)
        self.assertEqual(msg.delta.new_element.text.body, test_data)

class DeltaGeneratorContainerTest(DeltaGeneratorTestCase):
    """Test DeltaGenerator Container."""

    def test_container(self):
        if False:
            print('Hello World!')
        container = st.container()
        self.assertIsInstance(container, DeltaGenerator)
        self.assertFalse(container._cursor.is_locked)

    def test_container_paths(self):
        if False:
            i = 10
            return i + 15
        level3 = st.container().container().container()
        level3.markdown('hi')
        level3.markdown('bye')
        msg = self.get_message_from_queue()
        self.assertEqual(make_delta_path(RootContainer.MAIN, (0, 0, 0), 1), msg.metadata.delta_path)

class DeltaGeneratorColumnsTest(DeltaGeneratorTestCase):
    """Test DeltaGenerator Columns."""

    def test_equal_columns(self):
        if False:
            print('Hello World!')
        for column in st.columns(4):
            self.assertIsInstance(column, DeltaGenerator)
            self.assertFalse(column._cursor.is_locked)

    def test_variable_columns(self):
        if False:
            print('Hello World!')
        weights = [3, 1, 4, 1, 5, 9]
        sum_weights = sum(weights)
        st.columns(weights)
        for (i, w) in enumerate(weights):
            delta = self.get_delta_from_queue(i - len(weights))
            self.assertEqual(delta.add_block.column.weight, w / sum_weights)

    def test_bad_columns_negative_int(self):
        if False:
            print('Hello World!')
        with self.assertRaises(StreamlitAPIException):
            st.columns(-1337)

    def test_bad_columns_single_float(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            st.columns(6.28)

    def test_bad_columns_list_negative_value(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(StreamlitAPIException):
            st.columns([5, 6, -1.2])

    def test_bad_columns_list_int_zero_value(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(StreamlitAPIException):
            st.columns([5, 0, 1])

    def test_bad_columns_list_float_zero_value(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(StreamlitAPIException):
            st.columns([5.0, 0.0, 1.0])

    def test_two_levels_of_columns_does_not_raise_any_exception(self):
        if False:
            i = 10
            return i + 15
        (level1, _) = st.columns(2)
        try:
            (_, _) = level1.columns(2)
        except StreamlitAPIException:
            self.fail('Error, one level of nested columns should be allowed!')

    def test_three_levels_of_columns_raise_streamlit_api_exception(self):
        if False:
            print('Hello World!')
        (level1, _) = _ = st.columns(2)
        (level2, _) = level1.columns(2)
        exc = 'Columns can only be placed inside other columns up to one level of nesting.'
        with pytest.raises(StreamlitAPIException, match=exc):
            (_, _) = level2.columns(2)

    def test_one_level_of_columns_is_allowed_in_the_sidebar(self):
        if False:
            return 10
        try:
            with st.sidebar:
                (_, _) = st.columns(2)
        except StreamlitAPIException:
            self.fail('Error, 1 level column should be allowed in the sidebar!')

    def test_two_levels_of_columns_in_the_sidebar_raise_streamlit_api_exception(self):
        if False:
            for i in range(10):
                print('nop')
        exc = 'Columns cannot be placed inside other columns in the sidebar. This is only possible in the main area of the app.'
        with pytest.raises(StreamlitAPIException, match=exc):
            with st.sidebar:
                (col1, _) = st.columns(2)
                (_, _) = col1.columns(2)

class DeltaGeneratorExpanderTest(DeltaGeneratorTestCase):

    def test_nested_expanders(self):
        if False:
            while True:
                i = 10
        level1 = st.expander('level 1')
        with self.assertRaises(StreamlitAPIException):
            level1.expander('level 2')

class DeltaGeneratorWithTest(DeltaGeneratorTestCase):
    """Test the `with DG` feature"""

    def test_with(self):
        if False:
            return 10
        level3 = st.container().container().container()
        with level3:
            st.markdown('hi')
            st.markdown('bye')
        msg = self.get_message_from_queue()
        self.assertEqual(make_delta_path(RootContainer.MAIN, (0, 0, 0), 1), msg.metadata.delta_path)
        st.markdown('outside')
        msg = self.get_message_from_queue()
        self.assertEqual(make_delta_path(RootContainer.MAIN, (), 1), msg.metadata.delta_path)

    def test_nested_with(self):
        if False:
            return 10
        with st.container():
            with st.container():
                st.markdown('Level 2 with')
                msg = self.get_message_from_queue()
                self.assertEqual(make_delta_path(RootContainer.MAIN, (0, 0), 0), msg.metadata.delta_path)
            st.markdown('Level 1 with')
            msg = self.get_message_from_queue()
            self.assertEqual(make_delta_path(RootContainer.MAIN, (0,), 1), msg.metadata.delta_path)

class DeltaGeneratorWriteTest(DeltaGeneratorTestCase):
    """Test DeltaGenerator Text, Alert, Json, and Markdown Classes."""

    def test_json_list(self):
        if False:
            return 10
        'Test Text.JSON list.'
        json_data = [5, 6, 7, 8]
        st.json(json_data)
        json_string = json.dumps(json_data)
        element = self.get_delta_from_queue().new_element
        self.assertEqual(json_string, element.json.body)

    def test_json_tuple(self):
        if False:
            return 10
        'Test Text.JSON tuple.'
        json_data = (5, 6, 7, 8)
        st.json(json_data)
        json_string = json.dumps(json_data)
        element = self.get_delta_from_queue().new_element
        self.assertEqual(json_string, element.json.body)

    def test_json_object(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Text.JSON object.'
        json_data = {'key': 'value'}
        st.json(json_data)
        json_string = json.dumps(json_data)
        element = self.get_delta_from_queue().new_element
        self.assertEqual(json_string, element.json.body)
        self.assertEqual(True, element.json.expanded)

    def test_json_string(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Text.JSON string.'
        json_string = '{"key": "value"}'
        st.json(json_string)
        element = self.get_delta_from_queue().new_element
        self.assertEqual(json_string, element.json.body)

    def test_json_unserializable(self):
        if False:
            i = 10
            return i + 15
        'Test Text.JSON with unserializable object.'
        obj = json
        st.json(obj)
        element = self.get_delta_from_queue().new_element
        self.assertTrue(element.json.body.startswith('"<module \'json\''))

    def test_json_not_expanded_arg(self):
        if False:
            i = 10
            return i + 15
        'Test st.json expanded arg.'
        json_data = {'key': 'value'}
        st.json(json_data, expanded=False)
        json_string = json.dumps(json_data)
        element = self.get_delta_from_queue().new_element
        self.assertEqual(json_string, element.json.body)
        self.assertEqual(False, element.json.expanded)

    def test_json_not_mutates_data_containing_sets(self):
        if False:
            print('Hello World!')
        "Test st.json do not mutate data containing sets,\n        pass a dict-containing-a-set to st.json; ensure that it's not mutated\n        "
        json_data = {'some_set': {'a', 'b'}}
        self.assertIsInstance(json_data['some_set'], set)
        st.json(json_data)
        self.assertIsInstance(json_data['some_set'], set)

    def test_st_json_set_is_serialized_as_list(self):
        if False:
            i = 10
            return i + 15
        'Test st.json serializes set as list'
        json_data = {'a', 'b', 'c', 'd'}
        st.json(json_data)
        element = self.get_delta_from_queue().new_element
        parsed_element = json.loads(element.json.body)
        self.assertIsInstance(parsed_element, list)
        for el in json_data:
            self.assertIn(el, parsed_element)

    def test_st_json_serializes_sets_inside_iterables_as_lists(self):
        if False:
            for i in range(10):
                print('nop')
        'Test st.json serializes sets inside iterables as lists'
        json_data = {'some_set': {'a', 'b'}}
        st.json(json_data)
        element = self.get_delta_from_queue().new_element
        parsed_element = json.loads(element.json.body)
        set_as_list = parsed_element.get('some_set')
        self.assertIsInstance(set_as_list, list)
        self.assertSetEqual(json_data['some_set'], set(set_as_list))

    def test_st_json_generator_is_serialized_as_string(self):
        if False:
            return 10
        'Test st.json serializes generator as string'
        json_data = (c for c in 'foo')
        st.json(json_data)
        element = self.get_delta_from_queue().new_element
        parsed_element = json.loads(element.json.body)
        self.assertIsInstance(parsed_element, str)
        self.assertIn('generator', parsed_element)

    def test_markdown(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Markdown element.'
        test_string = '    data         '
        st.markdown(test_string)
        element = self.get_delta_from_queue().new_element
        self.assertEqual('data', element.markdown.body)
        test_string = '    <a#data>data</a>   '
        st.markdown(test_string)
        element = self.get_delta_from_queue().new_element
        assert element.markdown.body.startswith('<a#data>')

    def test_empty(self):
        if False:
            print('Hello World!')
        'Test Empty.'
        st.empty()
        element = self.get_delta_from_queue().new_element
        self.assertEqual(element.empty, EmptyProto())

class AutogeneratedWidgetIdTests(DeltaGeneratorTestCase):

    def test_ids_are_equal_when_inputs_are_equal(self):
        if False:
            i = 10
            return i + 15
        id1 = compute_widget_id('text_input', label='Label #1', default='Value #1')
        id2 = compute_widget_id('text_input', label='Label #1', default='Value #1')
        assert id1 == id2

    def test_ids_are_diff_when_labels_are_diff(self):
        if False:
            while True:
                i = 10
        id1 = compute_widget_id('text_input', label='Label #1', default='Value #1')
        id2 = compute_widget_id('text_input', label='Label #2', default='Value #1')
        assert id1 != id2

    def test_ids_are_diff_when_types_are_diff(self):
        if False:
            print('Hello World!')
        id1 = compute_widget_id('text_input', label='Label #1', default='Value #1')
        id2 = compute_widget_id('text_area', label='Label #1', default='Value #1')
        assert id1 != id2

class KeyWidgetIdTests(DeltaGeneratorTestCase):

    def test_ids_are_diff_when_keys_are_diff(self):
        if False:
            print('Hello World!')
        id1 = compute_widget_id('text_input', user_key='some_key1', label='Label #1', default='Value #1', key='some_key1')
        id2 = compute_widget_id('text_input', user_key='some_key2', label='Label #1', default='Value #1', key='some_key2')
        assert id1 != id2

class DeltaGeneratorImageTest(DeltaGeneratorTestCase):
    """Test DeltaGenerator Images"""

    def test_image_from_url(self):
        if False:
            while True:
                i = 10
        'Tests dg.image with single and multiple image URLs'
        url = 'https://streamlit.io/an_image.png'
        caption = 'ahoy!'
        st.image(url, caption=caption, width=200)
        element = self.get_delta_from_queue().new_element
        self.assertEqual(element.imgs.width, 200)
        self.assertEqual(len(element.imgs.imgs), 1)
        self.assertEqual(element.imgs.imgs[0].url, url)
        self.assertEqual(element.imgs.imgs[0].caption, caption)
        st.image([url] * 5, caption=[caption] * 5, width=200)
        element = self.get_delta_from_queue().new_element
        self.assertEqual(len(element.imgs.imgs), 5)
        self.assertEqual(element.imgs.imgs[4].url, url)
        self.assertEqual(element.imgs.imgs[4].caption, caption)

    def test_unequal_images_and_captions_error(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that the number of images and captions must match, or\n        an exception is generated'
        url = 'https://streamlit.io/an_image.png'
        caption = 'ahoy!'
        with self.assertRaises(Exception) as ctx:
            st.image([url] * 5, caption=[caption] * 2)
        self.assertTrue('Cannot pair 2 captions with 5 images.' in str(ctx.exception))