import datetime
import streamlit as st
from streamlit.errors import StreamlitAPIException
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class StreamlitAPITest(DeltaGeneratorTestCase):
    """Test Public Streamlit Public APIs."""

    def test_st_time_input(self):
        if False:
            while True:
                i = 10
        'Test st.time_input.'
        value = datetime.time(8, 45)
        st.time_input('Set an alarm for', value)
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.time_input.default, '08:45')
        self.assertEqual(el.time_input.step, datetime.timedelta(minutes=15).seconds)

    def test_st_time_input_with_step(self):
        if False:
            while True:
                i = 10
        'Test st.time_input with step.'
        value = datetime.time(9, 0)
        st.time_input('Set an alarm for', value, step=datetime.timedelta(minutes=5))
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.time_input.default, '09:00')
        self.assertEqual(el.time_input.step, datetime.timedelta(minutes=5).seconds)

    def test_st_time_input_exceptions(self):
        if False:
            for i in range(10):
                print('nop')
        'Test st.time_input exceptions.'
        value = datetime.time(9, 0)
        with self.assertRaises(StreamlitAPIException):
            st.time_input('Set an alarm for', value, step=True)
        with self.assertRaises(StreamlitAPIException):
            st.time_input('Set an alarm for', value, step=(90, 0))
        with self.assertRaises(StreamlitAPIException):
            st.time_input('Set an alarm for', value, step=1)
        with self.assertRaises(StreamlitAPIException):
            st.time_input('Set an alarm for', value, step=59)
        with self.assertRaises(StreamlitAPIException):
            st.time_input('Set an alarm for', value, step=datetime.timedelta(hours=24))
        with self.assertRaises(StreamlitAPIException):
            st.time_input('Set an alarm for', value, step=datetime.timedelta(days=1))