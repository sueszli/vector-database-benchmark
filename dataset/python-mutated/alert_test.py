from parameterized import parameterized
import streamlit as st
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Alert_pb2 import Alert
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class AlertAPITest(DeltaGeneratorTestCase):
    """Test ability to marshall Alert proto."""

    @parameterized.expand([(st.error,), (st.warning,), (st.info,), (st.success,)])
    def test_st_alert_exceptions(self, alert_func):
        if False:
            i = 10
            return i + 15
        'Test that alert functions throw an exception when a non-emoji is given as an icon.'
        with self.assertRaises(StreamlitAPIException):
            alert_func('some alert', icon='hello world')

class StErrorAPITest(DeltaGeneratorTestCase):
    """Test ability to marshall Alert proto."""

    def test_st_error(self):
        if False:
            print('Hello World!')
        'Test st.error.'
        st.error('some error')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.alert.body, 'some error')
        self.assertEqual(el.alert.format, Alert.ERROR)

    def test_st_error_with_icon(self):
        if False:
            return 10
        'Test st.error with icon.'
        st.error('some error', icon='😱')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.alert.body, 'some error')
        self.assertEqual(el.alert.icon, '😱')
        self.assertEqual(el.alert.format, Alert.ERROR)

class StInfoAPITest(DeltaGeneratorTestCase):
    """Test ability to marshall Alert proto."""

    def test_st_info(self):
        if False:
            return 10
        'Test st.info.'
        st.info('some info')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.alert.body, 'some info')
        self.assertEqual(el.alert.format, Alert.INFO)

    def test_st_info_with_icon(self):
        if False:
            while True:
                i = 10
        'Test st.info with icon.'
        st.info('some info', icon='👉🏻')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.alert.body, 'some info')
        self.assertEqual(el.alert.icon, '👉🏻')
        self.assertEqual(el.alert.format, Alert.INFO)

class StSuccessAPITest(DeltaGeneratorTestCase):
    """Test ability to marshall Alert proto."""

    def test_st_success(self):
        if False:
            while True:
                i = 10
        'Test st.success.'
        st.success('some success')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.alert.body, 'some success')
        self.assertEqual(el.alert.format, Alert.SUCCESS)

    def test_st_success_with_icon(self):
        if False:
            while True:
                i = 10
        'Test st.success with icon.'
        st.success('some success', icon='✅')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.alert.body, 'some success')
        self.assertEqual(el.alert.icon, '✅')
        self.assertEqual(el.alert.format, Alert.SUCCESS)

class StWarningAPITest(DeltaGeneratorTestCase):
    """Test ability to marshall Alert proto."""

    def test_st_warning(self):
        if False:
            i = 10
            return i + 15
        'Test st.warning.'
        st.warning('some warning')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.alert.body, 'some warning')
        self.assertEqual(el.alert.format, Alert.WARNING)

    def test_st_warning_with_icon(self):
        if False:
            return 10
        'Test st.warning with icon.'
        st.warning('some warning', icon='⚠️')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.alert.body, 'some warning')
        self.assertEqual(el.alert.icon, '⚠️')
        self.assertEqual(el.alert.format, Alert.WARNING)