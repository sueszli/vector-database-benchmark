"""link_button unit t Nest."""
import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class LinkButtonTest(DeltaGeneratorTestCase):
    """Test ability to marshall link_button protos."""

    def test_just_label(self):
        if False:
            return 10
        'Test that it can be called with label and string or bytes data.'
        st.link_button('the label', url='https://streamlit.io')
        c = self.get_delta_from_queue().new_element.link_button
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.type, 'secondary')
        self.assertEqual(c.disabled, False)

    def test_just_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that it can be called with disabled param.'
        st.link_button('the label', url='https://streamlit.io', disabled=True)
        c = self.get_delta_from_queue().new_element.link_button
        self.assertEqual(c.disabled, True)

    def test_url_exist(self):
        if False:
            i = 10
            return i + 15
        'Test that file url exist in proto.'
        st.link_button('the label', url='https://streamlit.io')
        c = self.get_delta_from_queue().new_element.link_button
        self.assertTrue('https://streamlit.io' in c.url)

    def test_type(self):
        if False:
            i = 10
            return i + 15
        'Test that it can be called with type param.'
        st.link_button('the label', url='https://streamlit.io', type='primary')
        c = self.get_delta_from_queue().new_element.link_button
        self.assertEqual(c.type, 'primary')

    def test_use_container_width_can_be_set_to_true(self):
        if False:
            print('Hello World!')
        'Test use_container_width can be set to true.'
        st.link_button('label', url='https://streamlit.io', use_container_width=True)
        c = self.get_delta_from_queue().new_element.link_button
        self.assertEqual(c.use_container_width, True)

    def test_use_container_width_is_false_by_default(self):
        if False:
            return 10
        'Test use_container_width is false by default.'
        st.link_button('the label', url='https://streamlit.io')
        c = self.get_delta_from_queue().new_element.link_button
        self.assertEqual(c.use_container_width, False)