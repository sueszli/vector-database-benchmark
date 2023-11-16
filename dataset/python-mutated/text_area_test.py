"""text_area unit test."""
import re
from unittest.mock import MagicMock, patch
from parameterized import parameterized
import streamlit as st
from streamlit.errors import StreamlitAPIException
from streamlit.proto.LabelVisibilityMessage_pb2 import LabelVisibilityMessage
from streamlit.testing.v1.app_test import AppTest
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class TextAreaTest(DeltaGeneratorTestCase):
    """Test ability to marshall text_area protos."""

    def test_just_label(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that it can be called with no value.'
        st.text_area('the label')
        c = self.get_delta_from_queue().new_element.text_area
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.label_visibility.value, LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE)
        self.assertEqual(c.default, '')
        self.assertEqual(c.HasField('default'), True)
        self.assertEqual(c.disabled, False)

    def test_just_disabled(self):
        if False:
            return 10
        'Test that it can be called with disabled param.'
        st.text_area('the label', disabled=True)
        c = self.get_delta_from_queue().new_element.text_area
        self.assertEqual(c.disabled, True)

    def test_value_types(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that it supports different types of values.'
        arg_values = ['some str', 123, {}, SomeObj()]
        proto_values = ['some str', '123', '{}', '.*SomeObj.*']
        for (arg_value, proto_value) in zip(arg_values, proto_values):
            st.text_area('the label', arg_value)
            c = self.get_delta_from_queue().new_element.text_area
            self.assertEqual(c.label, 'the label')
            self.assertTrue(re.match(proto_value, c.default))

    def test_none_value(self):
        if False:
            return 10
        'Test that it can be called with None as initial value.'
        st.text_area('the label', value=None)
        c = self.get_delta_from_queue().new_element.text_area
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, '')
        self.assertEqual(c.HasField('default'), False)

    def test_height(self):
        if False:
            i = 10
            return i + 15
        'Test that it can be called with height'
        st.text_area('the label', '', 300)
        c = self.get_delta_from_queue().new_element.text_area
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, '')
        self.assertEqual(c.height, 300)

    def test_placeholder(self):
        if False:
            while True:
                i = 10
        'Test that it can be called with placeholder'
        st.text_area('the label', '', placeholder='testing')
        c = self.get_delta_from_queue().new_element.text_area
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, '')
        self.assertEqual(c.placeholder, 'testing')

    def test_outside_form(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that form id is marshalled correctly outside of a form.'
        st.text_area('foo')
        proto = self.get_delta_from_queue().new_element.color_picker
        self.assertEqual(proto.form_id, '')

    @patch('streamlit.runtime.Runtime.exists', MagicMock(return_value=True))
    def test_inside_form(self):
        if False:
            while True:
                i = 10
        'Test that form id is marshalled correctly inside of a form.'
        with st.form('form'):
            st.text_area('foo')
        self.assertEqual(len(self.get_all_deltas_from_queue()), 2)
        form_proto = self.get_delta_from_queue(0).add_block
        text_area_proto = self.get_delta_from_queue(1).new_element.text_area
        self.assertEqual(text_area_proto.form_id, form_proto.form.form_id)

    def test_inside_column(self):
        if False:
            while True:
                i = 10
        'Test that it works correctly inside of a column.'
        (col1, col2, col3) = st.columns([2.5, 1.5, 8.3])
        with col1:
            st.text_area('foo')
        all_deltas = self.get_all_deltas_from_queue()
        self.assertEqual(len(all_deltas), 5)
        text_area_proto = self.get_delta_from_queue().new_element.text_area
        self.assertEqual(text_area_proto.label, 'foo')

    @parameterized.expand([('visible', LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE), ('hidden', LabelVisibilityMessage.LabelVisibilityOptions.HIDDEN), ('collapsed', LabelVisibilityMessage.LabelVisibilityOptions.COLLAPSED)])
    def test_label_visibility(self, label_visibility_value, proto_value):
        if False:
            print('Hello World!')
        'Test that it can be called with label_visibility param.'
        st.text_area('the label', label_visibility=label_visibility_value)
        c = self.get_delta_from_queue().new_element.text_area
        self.assertEqual(c.label_visibility.value, proto_value)

    def test_label_visibility_wrong_value(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(StreamlitAPIException) as e:
            st.text_area('the label', label_visibility='wrong_value')
        self.assertEqual(str(e.exception), "Unsupported label_visibility option 'wrong_value'. Valid values are 'visible', 'hidden' or 'collapsed'.")

    def test_help_dedents(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that help properly dedents'
        st.text_area('the label', value='TESTING', help='        Hello World!\n        This is a test\n\n\n        ')
        c = self.get_delta_from_queue().new_element.text_area
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 'TESTING')
        self.assertEqual(c.help, 'Hello World!\nThis is a test\n\n\n')

class SomeObj(object):
    pass

def test_text_input_interaction():
    if False:
        i = 10
        return i + 15
    'Test interactions with an empty text_area widget.'

    def script():
        if False:
            i = 10
            return i + 15
        import streamlit as st
        st.text_area('the label', value=None)
    at = AppTest.from_function(script).run()
    text_area = at.text_area[0]
    assert text_area.value is None
    at = text_area.input('Foo').run()
    text_area = at.text_area[0]
    assert text_area.value == 'Foo'
    at = text_area.set_value(None).run()
    text_area = at.text_area[0]
    assert text_area.value is None