"""number_input unit test."""
from unittest.mock import MagicMock, patch
import pytest
from parameterized import parameterized
import streamlit as st
from streamlit.errors import StreamlitAPIException
from streamlit.js_number import JSNumber
from streamlit.proto.Alert_pb2 import Alert as AlertProto
from streamlit.proto.LabelVisibilityMessage_pb2 import LabelVisibilityMessage
from streamlit.proto.NumberInput_pb2 import NumberInput
from streamlit.proto.WidgetStates_pb2 import WidgetState
from streamlit.testing.v1.app_test import AppTest
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class NumberInputTest(DeltaGeneratorTestCase):

    def test_data_type(self):
        if False:
            return 10
        'Test that NumberInput.type is set to the proper\n        NumberInput.DataType value\n        '
        st.number_input('Label', value=0)
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(NumberInput.INT, c.data_type)
        st.number_input('Label', value=0.5)
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(NumberInput.FLOAT, c.data_type)

    def test_min_value_zero_sets_default_value(self):
        if False:
            i = 10
            return i + 15
        st.number_input('Label', 0, 10)
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.default, 0)

    def test_just_label(self):
        if False:
            return 10
        'Test that it can be called with no value.'
        st.number_input('the label')
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.label_visibility.value, LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE)
        self.assertEqual(c.default, 0.0)
        self.assertEqual(c.HasField('default'), True)
        self.assertEqual(c.has_min, False)
        self.assertEqual(c.has_max, False)
        self.assertEqual(c.disabled, False)
        self.assertEqual(c.placeholder, '')

    def test_just_disabled(self):
        if False:
            print('Hello World!')
        'Test that it can be called with disabled param.'
        st.number_input('the label', disabled=True)
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.disabled, True)

    def test_placeholder(self):
        if False:
            while True:
                i = 10
        'Test that it can be called with placeholder param.'
        st.number_input('the label', placeholder='Type a number...')
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.placeholder, 'Type a number...')

    def test_none_value(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that it can be called with None as value.'
        st.number_input('the label', value=None)
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 0.0)
        self.assertEqual(c.HasField('default'), False)

    def test_none_value_with_int_min(self):
        if False:
            print('Hello World!')
        'Test that it can be called with None as value and\n        will be interpreted as integer if min_value is set to int.'
        st.number_input('the label', value=None, min_value=1)
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 0.0)
        self.assertEqual(c.HasField('default'), False)
        self.assertEqual(c.has_min, True)
        self.assertEqual(c.min, 1)
        self.assertEqual(c.data_type, NumberInput.INT)

    def test_default_value_when_min_is_passed(self):
        if False:
            print('Hello World!')
        st.number_input('the label', min_value=1, max_value=10)
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 1)

    def test_value_between_range(self):
        if False:
            while True:
                i = 10
        st.number_input('the label', 0, 11, 10)
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 10)
        self.assertEqual(c.min, 0)
        self.assertEqual(c.max, 11)
        self.assertEqual(c.has_min, True)
        self.assertEqual(c.has_max, True)

    def test_default_step_when_a_value_is_int(self):
        if False:
            for i in range(10):
                print('nop')
        st.number_input('the label', value=10)
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.step, 1.0)

    def test_default_step_when_a_value_is_float(self):
        if False:
            i = 10
            return i + 15
        st.number_input('the label', value=10.5)
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual('%0.2f' % c.step, '0.01')

    def test_default_format_int(self):
        if False:
            i = 10
            return i + 15
        st.number_input('the label', value=10)
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.format, '%d')

    def test_default_format_float(self):
        if False:
            while True:
                i = 10
        st.number_input('the label', value=10.5)
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.format, '%0.2f')

    def test_format_int_and_default_step(self):
        if False:
            for i in range(10):
                print('nop')
        st.number_input('the label', value=10, format='%d')
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.format, '%d')
        self.assertEqual(c.step, 1)

    def test_format_float_and_default_step(self):
        if False:
            while True:
                i = 10
        st.number_input('the label', value=10.0, format='%f')
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.format, '%f')
        self.assertEqual('%0.2f' % c.step, '0.01')

    def test_accept_valid_formats(self):
        if False:
            return 10
        SUPPORTED = 'adifFeEgGuXxo'
        for char in SUPPORTED:
            st.number_input('any label', format='%' + char)
            c = self.get_delta_from_queue().new_element.number_input
            self.assertEqual(c.format, '%' + char)

    def test_warns_on_float_type_with_int_format(self):
        if False:
            i = 10
            return i + 15
        st.number_input('the label', value=5.0, format='%d')
        c = self.get_delta_from_queue(-2).new_element.alert
        self.assertEqual(c.format, AlertProto.WARNING)
        self.assertEqual(c.body, 'Warning: NumberInput value below has type float, but format %d displays as integer.')

    def test_warns_on_int_type_with_float_format(self):
        if False:
            return 10
        st.number_input('the label', value=5, format='%0.2f')
        c = self.get_delta_from_queue(-2).new_element.alert
        self.assertEqual(c.format, AlertProto.WARNING)
        self.assertEqual(c.body, 'Warning: NumberInput value below has type int so is displayed as int despite format string %0.2f.')

    def test_error_on_unsupported_formatters(self):
        if False:
            for i in range(10):
                print('nop')
        UNSUPPORTED = 'pAn'
        for char in UNSUPPORTED:
            with pytest.raises(StreamlitAPIException) as exc_message:
                st.number_input('any label', value=3.14, format='%' + char)

    def test_error_on_invalid_formats(self):
        if False:
            i = 10
            return i + 15
        BAD_FORMATS = ['blah', 'a%f', 'a%.3f', '%d%d']
        for fmt in BAD_FORMATS:
            with pytest.raises(StreamlitAPIException) as exc_message:
                st.number_input('any label', value=3.14, format=fmt)

    def test_value_out_of_bounds(self):
        if False:
            print('Hello World!')
        with pytest.raises(StreamlitAPIException) as exc:
            value = JSNumber.MAX_SAFE_INTEGER + 1
            st.number_input('Label', value=value)
        self.assertEqual('`value` (%s) must be <= (1 << 53) - 1' % str(value), str(exc.value))
        with pytest.raises(StreamlitAPIException) as exc:
            value = JSNumber.MIN_SAFE_INTEGER - 1
            st.number_input('Label', value=value)
        self.assertEqual('`value` (%s) must be >= -((1 << 53) - 1)' % str(value), str(exc.value))
        with pytest.raises(StreamlitAPIException) as exc:
            value = 1e309
            st.number_input('Label', value=value)
        self.assertEqual('`value` (%s) must be <= 1.797e+308' % str(value), str(exc.value))
        with pytest.raises(StreamlitAPIException) as exc:
            value = -1e309
            st.number_input('Label', value=value)
        self.assertEqual('`value` (%s) must be >= -1.797e+308' % str(value), str(exc.value))

    def test_outside_form(self):
        if False:
            i = 10
            return i + 15
        'Test that form id is marshalled correctly outside of a form.'
        st.number_input('foo')
        proto = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(proto.form_id, '')

    @patch('streamlit.runtime.Runtime.exists', MagicMock(return_value=True))
    def test_inside_form(self):
        if False:
            print('Hello World!')
        'Test that form id is marshalled correctly inside of a form.'
        with st.form('form'):
            st.number_input('foo')
        self.assertEqual(len(self.get_all_deltas_from_queue()), 2)
        form_proto = self.get_delta_from_queue(0).add_block
        number_input_proto = self.get_delta_from_queue(1).new_element.number_input
        self.assertEqual(number_input_proto.form_id, form_proto.form.form_id)

    def test_inside_column(self):
        if False:
            while True:
                i = 10
        'Test that it works correctly inside of a column.'
        (col1, col2) = st.columns(2)
        with col1:
            st.number_input('foo', 0, 10)
        all_deltas = self.get_all_deltas_from_queue()
        self.assertEqual(len(all_deltas), 4)
        number_input_proto = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(number_input_proto.label, 'foo')
        self.assertEqual(number_input_proto.step, 1.0)
        self.assertEqual(number_input_proto.default, 0)

    @patch('streamlit.runtime.Runtime.exists', MagicMock(return_value=True))
    @patch('streamlit.elements.utils.get_session_state')
    def test_no_warning_with_value_set_in_state(self, patched_get_session_state):
        if False:
            print('Hello World!')
        mock_session_state = MagicMock()
        mock_session_state.is_new_state_value.return_value = True
        patched_get_session_state.return_value = mock_session_state
        st.number_input('the label', min_value=1, max_value=10, key='number_input')
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, 1)
        self.assertEqual(len(self.get_all_deltas_from_queue()), 1)

    @parameterized.expand([('visible', LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE), ('hidden', LabelVisibilityMessage.LabelVisibilityOptions.HIDDEN), ('collapsed', LabelVisibilityMessage.LabelVisibilityOptions.COLLAPSED)])
    def test_label_visibility(self, label_visibility_value, proto_value):
        if False:
            for i in range(10):
                print('nop')
        'Test that it can be called with label_visibility param.'
        st.number_input('the label', label_visibility=label_visibility_value)
        c = self.get_delta_from_queue().new_element.number_input
        self.assertEqual(c.label_visibility.value, proto_value)

    def test_label_visibility_wrong_value(self):
        if False:
            return 10
        with self.assertRaises(StreamlitAPIException) as e:
            st.number_input('the label', label_visibility='wrong_value')
        self.assertEqual(str(e.exception), "Unsupported label_visibility option 'wrong_value'. Valid values are 'visible', 'hidden' or 'collapsed'.")

    def test_should_keep_type_of_return_value_after_rerun(self):
        if False:
            return 10
        st.number_input('a number', min_value=1, max_value=100, key='number')
        widget_id = self.script_run_ctx.session_state.get_widget_states()[0].id
        self.script_run_ctx.reset()
        widget_state = WidgetState()
        widget_state.id = widget_id
        widget_state.double_value = 42.0
        self.script_run_ctx.session_state._state._new_widget_state.set_widget_from_proto(widget_state)
        number = st.number_input('a number', min_value=1, max_value=100, key='number')
        self.assertEqual(number, 42)
        self.assertEqual(type(number), int)

    @parameterized.expand([(6, -10, 0), (-11, -10, 0), (-11.0, -10.0, 0.0), (6.0, -10.0, 0.0)])
    def test_should_raise_exception_when_default_out_of_bounds_min_and_max_defined(self, value, min_value, max_value):
        if False:
            i = 10
            return i + 15
        with pytest.raises(StreamlitAPIException):
            st.number_input('My Label', value=value, min_value=min_value, max_value=max_value)

    def test_should_raise_exception_when_default_lt_min_and_max_is_none(self):
        if False:
            return 10
        value = -11.0
        min_value = -10.0
        with pytest.raises(StreamlitAPIException):
            st.number_input('My Label', value=value, min_value=min_value)

    def test_should_raise_exception_when_default_gt_max_and_min_is_none(self):
        if False:
            while True:
                i = 10
        value = 11
        max_value = 10
        with pytest.raises(StreamlitAPIException):
            st.number_input('My Label', value=value, max_value=max_value)

def test_number_input_interaction():
    if False:
        for i in range(10):
            print('nop')
    'Test interactions with an empty number input widget.'

    def script():
        if False:
            for i in range(10):
                print('nop')
        import streamlit as st
        st.number_input('the label', value=None)
    at = AppTest.from_function(script).run()
    number_input = at.number_input[0]
    assert number_input.value is None
    at = number_input.set_value(10).run()
    number_input = at.number_input[0]
    assert number_input.value == 10
    at = number_input.increment().run()
    number_input = at.number_input[0]
    assert number_input.value == 10.01
    at = number_input.set_value(None).run()
    number_input = at.number_input[0]
    assert number_input.value is None