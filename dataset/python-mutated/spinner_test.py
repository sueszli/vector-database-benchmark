import time
from streamlit.elements.spinner import spinner
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class SpinnerTest(DeltaGeneratorTestCase):

    def test_spinner(self):
        if False:
            print('Hello World!')
        'Test st.spinner.'
        with spinner('some text'):
            time.sleep(0.7)
            el = self.get_delta_from_queue().new_element
            self.assertEqual(el.spinner.text, 'some text')
            self.assertFalse(el.spinner.cache)
        last_delta = self.get_delta_from_queue()
        self.assertTrue(last_delta.HasField('new_element'))
        self.assertEqual(last_delta.new_element.WhichOneof('type'), 'empty')

    def test_spinner_within_chat_message(self):
        if False:
            while True:
                i = 10
        'Test st.spinner in st.chat_message resets to empty container block.'
        import streamlit as st
        with st.chat_message('user'):
            with spinner('some text'):
                time.sleep(0.7)
                el = self.get_delta_from_queue().new_element
                self.assertEqual(el.spinner.text, 'some text')
                self.assertFalse(el.spinner.cache)
        last_delta = self.get_delta_from_queue()
        self.assertTrue(last_delta.HasField('add_block'))
        self.assertFalse(last_delta.add_block.allow_empty)

    def test_spinner_for_caching(self):
        if False:
            i = 10
            return i + 15
        'Test st.spinner in cache functions.'
        with spinner('some text', cache=True):
            time.sleep(0.7)
            el = self.get_delta_from_queue().new_element
            self.assertEqual(el.spinner.text, 'some text')
            self.assertTrue(el.spinner.cache)
        last_delta = self.get_delta_from_queue()
        self.assertTrue(last_delta.HasField('new_element'))
        self.assertEqual(last_delta.new_element.WhichOneof('type'), 'empty')