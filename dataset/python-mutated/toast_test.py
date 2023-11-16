"""toast unit tests."""
import streamlit as st
from streamlit.errors import StreamlitAPIException
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class ToastTest(DeltaGeneratorTestCase):

    def test_just_text(self):
        if False:
            print('Hello World!')
        'Test that it can be called with just text.'
        st.toast('toast text')
        c = self.get_delta_from_queue().new_element.toast
        self.assertEqual(c.body, 'toast text')
        self.assertEqual(c.icon, '')

    def test_no_text(self):
        if False:
            while True:
                i = 10
        'Test that an error is raised if no text is provided.'
        with self.assertRaises(StreamlitAPIException) as e:
            st.toast('')
        self.assertEqual(str(e.exception), 'Toast body cannot be blank - please provide a message.')

    def test_valid_icon(self):
        if False:
            while True:
                i = 10
        'Test that it can be called passing a valid emoji as icon.'
        st.toast('toast text', icon='🦄')
        c = self.get_delta_from_queue().new_element.toast
        self.assertEqual(c.body, 'toast text')
        self.assertEqual(c.icon, '🦄')

    def test_invalid_icon(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that an error is raised if an invalid icon is provided.'
        with self.assertRaises(StreamlitAPIException) as e:
            st.toast('toast text', icon='invalid')
        self.assertEqual(str(e.exception), 'The value "invalid" is not a valid emoji. Shortcodes are not allowed, please use a single character instead.')