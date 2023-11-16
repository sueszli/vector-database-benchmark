"""radio unit tests."""
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from parameterized import parameterized
import streamlit as st
from streamlit.errors import StreamlitAPIException
from streamlit.proto.LabelVisibilityMessage_pb2 import LabelVisibilityMessage
from streamlit.testing.v1.app_test import AppTest
from streamlit.testing.v1.util import patch_config_options
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class RadioTest(DeltaGeneratorTestCase):
    """Test ability to marshall radio protos."""

    def test_just_label(self):
        if False:
            print('Hello World!')
        'Test that it can be called with no value.'
        st.radio('the label', ('m', 'f'))
        c = self.get_delta_from_queue().new_element.radio
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.label_visibility.value, LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE)
        self.assertEqual(c.default, 0)
        self.assertEqual(c.disabled, False)
        self.assertEqual(c.HasField('default'), True)
        self.assertEqual(c.captions, [])

    def test_just_disabled(self):
        if False:
            while True:
                i = 10
        'Test that it can be called with disabled param.'
        st.radio('the label', ('m', 'f'), disabled=True)
        c = self.get_delta_from_queue().new_element.radio
        self.assertEqual(c.disabled, True)

    def test_none_value(self):
        if False:
            i = 10
            return i + 15
        'Test that it can be called with None as index value.'
        st.radio('the label', ('m', 'f'), index=None)
        c = self.get_delta_from_queue().new_element.radio
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 0)
        self.assertEqual(c.HasField('default'), False)

    def test_horizontal(self):
        if False:
            i = 10
            return i + 15
        'Test that it can be called with horizontal param.'
        st.radio('the label', ('m', 'f'), horizontal=True)
        c = self.get_delta_from_queue().new_element.radio
        self.assertEqual(c.horizontal, True)

    def test_horizontal_default_value(self):
        if False:
            i = 10
            return i + 15
        'Test that it can called with horizontal param value False by default.'
        st.radio('the label', ('m', 'f'))
        c = self.get_delta_from_queue().new_element.radio
        self.assertEqual(c.horizontal, False)

    def test_valid_value(self):
        if False:
            return 10
        'Test that valid value is an int.'
        st.radio('the label', ('m', 'f'), 1)
        c = self.get_delta_from_queue().new_element.radio
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 1)

    def test_noneType_option(self):
        if False:
            return 10
        'Test NoneType option value.'
        current_value = st.radio('the label', (None, 'selected'), 0)
        self.assertEqual(current_value, None)

    @parameterized.expand([(('m', 'f'), ['m', 'f']), (['male', 'female'], ['male', 'female']), (np.array(['m', 'f']), ['m', 'f']), (pd.Series(np.array(['male', 'female'])), ['male', 'female']), (pd.DataFrame({'options': ['male', 'female']}), ['male', 'female']), (pd.DataFrame(data=[[1, 4, 7], [2, 5, 8], [3, 6, 9]], columns=['a', 'b', 'c']).columns, ['a', 'b', 'c'])])
    def test_option_types(self, options, proto_options):
        if False:
            return 10
        'Test that it supports different types of options.'
        st.radio('the label', options)
        c = self.get_delta_from_queue().new_element.radio
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 0)
        self.assertEqual(c.options, proto_options)

    def test_cast_options_to_string(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that it casts options to string.'
        arg_options = ['some str', 123, None, {}]
        proto_options = ['some str', '123', 'None', '{}']
        st.radio('the label', arg_options)
        c = self.get_delta_from_queue().new_element.radio
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 0)
        self.assertEqual(c.options, proto_options)

    def test_format_function(self):
        if False:
            i = 10
            return i + 15
        'Test that it formats options.'
        arg_options = [{'name': 'john', 'height': 180}, {'name': 'lisa', 'height': 200}]
        proto_options = ['john', 'lisa']
        st.radio('the label', arg_options, format_func=lambda x: x['name'])
        c = self.get_delta_from_queue().new_element.radio
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 0)
        self.assertEqual(c.options, proto_options)

    @parameterized.expand([((),), ([],), (np.array([]),), (pd.Series(np.array([])),)])
    def test_no_options(self, options):
        if False:
            i = 10
            return i + 15
        'Test that it handles no options.'
        st.radio('the label', options)
        c = self.get_delta_from_queue().new_element.radio
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.label_visibility.value, LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE)
        self.assertEqual(c.default, 0)
        self.assertEqual(c.options, [])

    def test_invalid_value(self):
        if False:
            i = 10
            return i + 15
        'Test that value must be an int.'
        with self.assertRaises(StreamlitAPIException):
            st.radio('the label', ('m', 'f'), '1')

    def test_invalid_value_range(self):
        if False:
            i = 10
            return i + 15
        'Test that value must be within the length of the options.'
        with self.assertRaises(StreamlitAPIException):
            st.radio('the label', ('m', 'f'), 2)

    def test_outside_form(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that form id is marshalled correctly outside of a form.'
        st.radio('foo', ['bar', 'baz'])
        proto = self.get_delta_from_queue().new_element.radio
        self.assertEqual(proto.form_id, '')

    @patch('streamlit.runtime.Runtime.exists', MagicMock(return_value=True))
    def test_inside_form(self):
        if False:
            print('Hello World!')
        'Test that form id is marshalled correctly inside of a form.'
        with st.form('form'):
            st.radio('foo', ['bar', 'baz'])
        self.assertEqual(len(self.get_all_deltas_from_queue()), 2)
        form_proto = self.get_delta_from_queue(0).add_block
        radio_proto = self.get_delta_from_queue(1).new_element.radio
        self.assertEqual(radio_proto.form_id, form_proto.form.form_id)

    def test_inside_column(self):
        if False:
            return 10
        'Test that it works correctly inside of a column.'
        (col1, col2) = st.columns(2)
        with col1:
            st.radio('foo', ['bar', 'baz'])
        all_deltas = self.get_all_deltas_from_queue()
        self.assertEqual(len(all_deltas), 4)
        radio_proto = self.get_delta_from_queue().new_element.radio
        self.assertEqual(radio_proto.label, 'foo')
        self.assertEqual(radio_proto.options, ['bar', 'baz'])
        self.assertEqual(radio_proto.default, 0)

    @parameterized.expand([('visible', LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE), ('hidden', LabelVisibilityMessage.LabelVisibilityOptions.HIDDEN), ('collapsed', LabelVisibilityMessage.LabelVisibilityOptions.COLLAPSED)])
    def test_label_visibility(self, label_visibility_value, proto_value):
        if False:
            while True:
                i = 10
        'Test that it can be called with label_visibility param.'
        st.radio('the label', ('m', 'f'), label_visibility=label_visibility_value)
        c = self.get_delta_from_queue().new_element.radio
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 0)
        self.assertEqual(c.label_visibility.value, proto_value)

    def test_label_visibility_wrong_value(self):
        if False:
            return 10
        with self.assertRaises(StreamlitAPIException) as e:
            st.radio('the label', ('m', 'f'), label_visibility='wrong_value')
        self.assertEqual(str(e.exception), "Unsupported label_visibility option 'wrong_value'. Valid values are 'visible', 'hidden' or 'collapsed'.")

    def test_no_captions(self):
        if False:
            print('Hello World!')
        'Test that it can be called with no captions.'
        st.radio('the label', ('option1', 'option2', 'option3'), captions=None)
        c = self.get_delta_from_queue().new_element.radio
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 0)
        self.assertEqual(c.captions, [])

    def test_some_captions(self):
        if False:
            print('Hello World!')
        'Test that it can be called with some captions.'
        st.radio('the label', ('option1', 'option2', 'option3', 'option4'), captions=('first caption', None, '', 'last caption'))
        c = self.get_delta_from_queue().new_element.radio
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 0)
        self.assertEqual(c.captions, ['first caption', '', '', 'last caption'])

def test_radio_interaction():
    if False:
        print('Hello World!')
    'Test interactions with an empty radio widget.'

    def script():
        if False:
            print('Hello World!')
        import streamlit as st
        st.radio('the label', ('m', 'f'), index=None)
    at = AppTest.from_function(script).run()
    radio = at.radio[0]
    assert radio.value is None
    at = radio.set_value('m').run()
    radio = at.radio[0]
    assert radio.value == 'm'
    at = radio.set_value(None).run()
    radio = at.radio[0]
    assert radio.value is None

def test_radio_enum_coercion():
    if False:
        i = 10
        return i + 15
    'Test E2E Enum Coercion on a radio.'

    def script():
        if False:
            while True:
                i = 10
        from enum import Enum
        import streamlit as st

        class EnumA(Enum):
            A = 1
            B = 2
            C = 3
        selected = st.radio('my_enum', EnumA, index=0)
        st.text(id(selected.__class__))
        st.text(id(EnumA))
        st.text(selected in EnumA)
    at = AppTest.from_function(script).run()

    def test_enum():
        if False:
            i = 10
            return i + 15
        radio = at.radio[0]
        original_class = radio.value.__class__
        radio.set_value(original_class.C).run()
        assert at.text[0].value == at.text[1].value, 'Enum Class ID not the same'
        assert at.text[2].value == 'True', 'Not all enums found in class'
    with patch_config_options({'runner.enumCoercion': 'nameOnly'}):
        test_enum()
    with patch_config_options({'runner.enumCoercion': 'off'}):
        with pytest.raises(AssertionError):
            test_enum()