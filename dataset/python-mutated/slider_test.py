"""slider unit test."""
from datetime import date, datetime, time, timedelta, timezone
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from parameterized import parameterized
import streamlit as st
from streamlit.errors import StreamlitAPIException
from streamlit.js_number import JSNumber
from streamlit.proto.LabelVisibilityMessage_pb2 import LabelVisibilityMessage
from streamlit.testing.v1.app_test import AppTest
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class SliderTest(DeltaGeneratorTestCase):
    """Test ability to marshall slider protos."""

    def test_just_label(self):
        if False:
            print('Hello World!')
        'Test that it can be called with no value.'
        st.slider('the label')
        c = self.get_delta_from_queue().new_element.slider
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.label_visibility.value, LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE)
        self.assertEqual(c.default, [0])
        self.assertEqual(c.disabled, False)

    def test_just_disabled(self):
        if False:
            while True:
                i = 10
        'Test that it can be called with disabled param.'
        st.slider('the label', disabled=True)
        c = self.get_delta_from_queue().new_element.slider
        self.assertEqual(c.disabled, True)
    PST = timezone(timedelta(hours=-8), 'PST')
    AWARE_DT = datetime(2020, 1, 1, tzinfo=PST)
    AWARE_DT_END = datetime(2020, 1, 5, tzinfo=PST)
    AWARE_TIME = time(12, 0, tzinfo=PST)
    AWARE_TIME_END = time(21, 0, tzinfo=PST)
    AWARE_DT_MICROS = 1577836800000000
    AWARE_DT_END_MICROS = 1578182400000000
    AWARE_TIME_MICROS = 946728000000000
    AWARE_TIME_END_MICROS = 946760400000000

    @parameterized.expand([(1, [1], 1), ((0, 1), [0, 1], (0, 1)), ([0, 1], [0, 1], (0, 1)), (0.5, [0.5], 0.5), ((0.2, 0.5), [0.2, 0.5], (0.2, 0.5)), ([0.2, 0.5], [0.2, 0.5], (0.2, 0.5)), (np.int64(1), [1], 1), (np.int32(1), [1], 1), (np.single(0.5), [0.5], 0.5), (np.double(0.5), [0.5], 0.5), (AWARE_DT, [AWARE_DT_MICROS], AWARE_DT), ((AWARE_DT, AWARE_DT_END), [AWARE_DT_MICROS, AWARE_DT_END_MICROS], (AWARE_DT, AWARE_DT_END)), ([AWARE_DT, AWARE_DT_END], [AWARE_DT_MICROS, AWARE_DT_END_MICROS], (AWARE_DT, AWARE_DT_END)), (AWARE_TIME, [AWARE_TIME_MICROS], AWARE_TIME), ((AWARE_TIME, AWARE_TIME_END), [AWARE_TIME_MICROS, AWARE_TIME_END_MICROS], (AWARE_TIME, AWARE_TIME_END)), ([AWARE_TIME, AWARE_TIME_END], [AWARE_TIME_MICROS, AWARE_TIME_END_MICROS], (AWARE_TIME, AWARE_TIME_END))])
    def test_value_types(self, value, proto_value, return_value):
        if False:
            print('Hello World!')
        'Test that it supports different types of values.'
        ret = st.slider('the label', value=value)
        self.assertEqual(ret, return_value)
        c = self.get_delta_from_queue().new_element.slider
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, proto_value)

    @parameterized.expand(['5', 5j, b'5'])
    def test_invalid_types(self, value):
        if False:
            print('Hello World!')
        'Test that it rejects invalid types, specifically things that are *almost* numbers'
        with pytest.raises(StreamlitAPIException):
            st.slider('the label', value=value)

    @parameterized.expand([(1, 1, 1, 1), (np.int64(1), 1, 1, 1), (1, np.int64(1), 1, 1), (1, 1, np.int64(1), 1), (np.single(0.5), 0.5, 0.5, 0.5)])
    def test_matching_types(self, min_value, max_value, value, return_value):
        if False:
            print('Hello World!')
        'Test that NumPy types are seen as compatible with numerical Python types'
        ret = st.slider('the label', min_value=min_value, max_value=max_value, value=value)
        self.assertEqual(ret, return_value)
    NAIVE_DT = datetime(2020, 2, 1)
    NAIVE_DT_END = datetime(2020, 2, 4)
    NAIVE_TIME = time(6, 20, 34)
    NAIVE_TIME_END = time(20, 6, 43)
    DATE_START = date(2020, 4, 5)
    DATE_END = date(2020, 6, 6)

    @parameterized.expand([(NAIVE_DT, NAIVE_DT), ((NAIVE_DT, NAIVE_DT_END), (NAIVE_DT, NAIVE_DT_END)), ([NAIVE_DT, NAIVE_DT_END], (NAIVE_DT, NAIVE_DT_END)), (NAIVE_TIME, NAIVE_TIME), ((NAIVE_TIME, NAIVE_TIME_END), (NAIVE_TIME, NAIVE_TIME_END)), ([NAIVE_TIME, NAIVE_TIME_END], (NAIVE_TIME, NAIVE_TIME_END)), (DATE_START, DATE_START), ((DATE_START, DATE_END), (DATE_START, DATE_END)), ([DATE_START, DATE_END], (DATE_START, DATE_END))])
    def test_naive_timelikes(self, value, return_value):
        if False:
            for i in range(10):
                print('nop')
        "Ignore proto values (they change based on testing machine's timezone)"
        ret = st.slider('the label', value=value)
        c = self.get_delta_from_queue().new_element.slider
        self.assertEqual(ret, return_value)
        self.assertEqual(c.label, 'the label')

    def test_range_session_state(self):
        if False:
            while True:
                i = 10
        'Test a range set by session state.'
        state = st.session_state
        state['slider'] = [10, 20]
        slider = st.slider('select a range', min_value=0, max_value=100, key='slider')
        assert slider == [10, 20]

    def test_value_greater_than_min(self):
        if False:
            while True:
                i = 10
        ret = st.slider('Slider label', 10, 100, 0)
        c = self.get_delta_from_queue().new_element.slider
        self.assertEqual(ret, 0)
        self.assertEqual(c.min, 0)

    def test_value_smaller_than_max(self):
        if False:
            return 10
        ret = st.slider('Slider label', 10, 100, 101)
        c = self.get_delta_from_queue().new_element.slider
        self.assertEqual(ret, 101)
        self.assertEqual(c.max, 101)

    def test_max_min(self):
        if False:
            return 10
        ret = st.slider('Slider label', 101, 100, 101)
        c = self.get_delta_from_queue().new_element.slider
        (self.assertEqual(ret, 101),)
        self.assertEqual(c.min, 100)
        self.assertEqual(c.max, 101)

    def test_value_out_of_bounds(self):
        if False:
            while True:
                i = 10
        with pytest.raises(StreamlitAPIException) as exc:
            max_value = JSNumber.MAX_SAFE_INTEGER + 1
            st.slider('Label', max_value=max_value)
        self.assertEqual('`max_value` (%s) must be <= (1 << 53) - 1' % str(max_value), str(exc.value))
        with pytest.raises(StreamlitAPIException) as exc:
            min_value = JSNumber.MIN_SAFE_INTEGER - 1
            st.slider('Label', min_value=min_value)
        self.assertEqual('`min_value` (%s) must be >= -((1 << 53) - 1)' % str(min_value), str(exc.value))
        with pytest.raises(StreamlitAPIException) as exc:
            max_value = 1e309
            st.slider('Label', value=0.5, max_value=max_value)
        self.assertEqual('`max_value` (%s) must be <= 1.797e+308' % str(max_value), str(exc.value))
        with pytest.raises(StreamlitAPIException) as exc:
            min_value = -1e309
            st.slider('Label', value=0.5, min_value=min_value)
        self.assertEqual('`min_value` (%s) must be >= -1.797e+308' % str(min_value), str(exc.value))

    def test_step_zero(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(StreamlitAPIException) as exc:
            st.slider('Label', min_value=0, max_value=10, step=0)
        self.assertEqual('Slider components cannot be passed a `step` of 0.', str(exc.value))

    def test_outside_form(self):
        if False:
            print('Hello World!')
        'Test that form id is marshalled correctly outside of a form.'
        st.slider('foo')
        proto = self.get_delta_from_queue().new_element.slider
        self.assertEqual(proto.form_id, '')

    @patch('streamlit.runtime.Runtime.exists', MagicMock(return_value=True))
    def test_inside_form(self):
        if False:
            print('Hello World!')
        'Test that form id is marshalled correctly inside of a form.'
        with st.form('form'):
            st.slider('foo')
        self.assertEqual(len(self.get_all_deltas_from_queue()), 2)
        form_proto = self.get_delta_from_queue(0).add_block
        slider_proto = self.get_delta_from_queue(1).new_element.slider
        self.assertEqual(slider_proto.form_id, form_proto.form.form_id)

    def test_inside_column(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that it works correctly inside of a column.'
        (col1, col2) = st.columns(2)
        with col1:
            st.slider('foo')
        all_deltas = self.get_all_deltas_from_queue()
        self.assertEqual(len(all_deltas), 4)
        slider_proto = self.get_delta_from_queue().new_element.slider
        self.assertEqual(slider_proto.label, 'foo')

    @parameterized.expand([('visible', LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE), ('hidden', LabelVisibilityMessage.LabelVisibilityOptions.HIDDEN), ('collapsed', LabelVisibilityMessage.LabelVisibilityOptions.COLLAPSED)])
    def test_label_visibility(self, label_visibility_value, proto_value):
        if False:
            return 10
        'Test that it can be called with label_visibility param.'
        st.slider('the label', label_visibility=label_visibility_value)
        c = self.get_delta_from_queue().new_element.slider
        self.assertEqual(c.label_visibility.value, proto_value)

    def test_label_visibility_wrong_value(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(StreamlitAPIException) as e:
            st.slider('the label', label_visibility='wrong_value')
        self.assertEqual(str(e.exception), "Unsupported label_visibility option 'wrong_value'. Valid values are 'visible', 'hidden' or 'collapsed'.")

def test_id_stability():
    if False:
        i = 10
        return i + 15

    def script():
        if False:
            for i in range(10):
                print('nop')
        import streamlit as st
        st.slider('slider', key='slider')
    at = AppTest.from_function(script).run()
    s1 = at.slider[0]
    at = s1.set_value(5).run()
    s2 = at.slider[0]
    assert s1.id == s2.id