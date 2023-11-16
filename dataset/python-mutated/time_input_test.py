"""time_input unit test."""
from datetime import datetime, time
from parameterized import parameterized
import streamlit as st
from streamlit.errors import StreamlitAPIException
from streamlit.proto.LabelVisibilityMessage_pb2 import LabelVisibilityMessage
from streamlit.testing.v1.app_test import AppTest
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class TimeInputTest(DeltaGeneratorTestCase):
    """Test ability to marshall time_input protos."""

    def test_just_label(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that it can be called with no value.'
        st.time_input('the label')
        c = self.get_delta_from_queue().new_element.time_input
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.label_visibility.value, LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE)
        self.assertLessEqual(datetime.strptime(c.default, '%H:%M').time(), datetime.now().time())
        self.assertEqual(c.HasField('default'), True)
        self.assertEqual(c.disabled, False)

    def test_just_disabled(self):
        if False:
            print('Hello World!')
        'Test that it can be called with disabled param.'
        st.time_input('the label', disabled=True)
        c = self.get_delta_from_queue().new_element.time_input
        self.assertEqual(c.disabled, True)

    def test_none_value(self):
        if False:
            while True:
                i = 10
        'Test that it can be called with None as initial value.'
        st.time_input('the label', value=None)
        c = self.get_delta_from_queue().new_element.time_input
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, '')
        self.assertEqual(c.HasField('default'), False)

    @parameterized.expand([(time(8, 45), '08:45'), (datetime(2019, 7, 6, 21, 15), '21:15')])
    def test_value_types(self, arg_value, proto_value):
        if False:
            return 10
        'Test that it supports different types of values.'
        st.time_input('the label', arg_value)
        c = self.get_delta_from_queue().new_element.time_input
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, proto_value)

    def test_inside_column(self):
        if False:
            print('Hello World!')
        'Test that it works correctly inside of a column.'
        (col1, col2) = st.columns([3, 2])
        with col1:
            st.time_input('foo')
        all_deltas = self.get_all_deltas_from_queue()
        self.assertEqual(len(all_deltas), 4)
        time_input_proto = self.get_delta_from_queue().new_element.time_input
        self.assertEqual(time_input_proto.label, 'foo')

    @parameterized.expand([('visible', LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE), ('hidden', LabelVisibilityMessage.LabelVisibilityOptions.HIDDEN), ('collapsed', LabelVisibilityMessage.LabelVisibilityOptions.COLLAPSED)])
    def test_label_visibility(self, label_visibility_value, proto_value):
        if False:
            return 10
        'Test that it can be called with label_visibility param.'
        st.time_input('the label', label_visibility=label_visibility_value)
        c = self.get_delta_from_queue().new_element.time_input
        self.assertEqual(c.label_visibility.value, proto_value)

    def test_label_visibility_wrong_value(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(StreamlitAPIException) as e:
            st.time_input('the label', label_visibility='wrong_value')
        self.assertEqual(str(e.exception), "Unsupported label_visibility option 'wrong_value'. Valid values are 'visible', 'hidden' or 'collapsed'.")

def test_time_input_interaction():
    if False:
        return 10
    'Test interactions with an empty time_input widget.'

    def script():
        if False:
            return 10
        import streamlit as st
        st.time_input('the label', value=None)
    at = AppTest.from_function(script).run()
    time_input = at.time_input[0]
    assert time_input.value is None
    at = time_input.set_value(time(8, 45)).run()
    time_input = at.time_input[0]
    assert time_input.value == time(8, 45)
    at = time_input.set_value(None).run()
    time_input = at.time_input[0]
    assert time_input.value is None