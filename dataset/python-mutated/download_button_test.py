"""download_button unit test."""
from parameterized import parameterized
import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class DownloadButtonTest(DeltaGeneratorTestCase):
    """Test ability to marshall download_button protos."""

    @parameterized.expand([('hello world',), (b'byteshere',)])
    def test_just_label(self, data):
        if False:
            while True:
                i = 10
        'Test that it can be called with label and string or bytes data.'
        st.download_button('the label', data=data)
        c = self.get_delta_from_queue().new_element.download_button
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.type, 'secondary')
        self.assertEqual(c.disabled, False)

    def test_just_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that it can be called with disabled param.'
        st.download_button('the label', data='juststring', disabled=True)
        c = self.get_delta_from_queue().new_element.download_button
        self.assertEqual(c.disabled, True)

    def test_url_exist(self):
        if False:
            i = 10
            return i + 15
        'Test that file url exist in proto.'
        st.download_button('the label', data='juststring')
        c = self.get_delta_from_queue().new_element.download_button
        self.assertTrue('/media/' in c.url)

    def test_type(self):
        if False:
            while True:
                i = 10
        'Test that it can be called with type param.'
        st.download_button('the label', data='Streamlit', type='primary')
        c = self.get_delta_from_queue().new_element.download_button
        self.assertEqual(c.type, 'primary')

    def test_use_container_width_can_be_set_to_true(self):
        if False:
            for i in range(10):
                print('nop')
        'Test use_container_width can be set to true.'
        st.download_button('the label', data='juststring', use_container_width=True)
        c = self.get_delta_from_queue().new_element.download_button
        self.assertEqual(c.use_container_width, True)

    def test_use_container_width_is_false_by_default(self):
        if False:
            return 10
        'Test use_container_width is false by default.'
        st.download_button('the label', data='juststring')
        c = self.get_delta_from_queue().new_element.download_button
        self.assertEqual(c.use_container_width, False)