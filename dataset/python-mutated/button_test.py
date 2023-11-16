"""button unit test."""
import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class ButtonTest(DeltaGeneratorTestCase):
    """Test ability to marshall button protos."""

    def test_button(self):
        if False:
            return 10
        'Test that it can be called.'
        st.button('the label')
        c = self.get_delta_from_queue().new_element.button
        self.assertEqual(c.label, 'the label')
        self.assertEqual(c.default, False)
        self.assertEqual(c.form_id, '')
        self.assertEqual(c.type, 'secondary')
        self.assertEqual(c.is_form_submitter, False)
        self.assertEqual(c.disabled, False)

    def test_type(self):
        if False:
            i = 10
            return i + 15
        'Test that it can be called with type param.'
        st.button('the label', type='primary')
        c = self.get_delta_from_queue().new_element.button
        self.assertEqual(c.type, 'primary')

    def test_just_disabled(self):
        if False:
            while True:
                i = 10
        'Test that it can be called with disabled param.'
        st.button('the label', disabled=True)
        c = self.get_delta_from_queue().new_element.button
        self.assertEqual(c.disabled, True)

    def test_use_container_width_can_be_set_to_true(self):
        if False:
            i = 10
            return i + 15
        'Test use_container_width can be set to true.'
        st.button('the label', use_container_width=True)
        c = self.get_delta_from_queue().new_element.button
        self.assertEqual(c.use_container_width, True)

    def test_use_container_width_is_false_by_default(self):
        if False:
            i = 10
            return i + 15
        'Test use_container_width is false by default.'
        st.button('the label')
        c = self.get_delta_from_queue().new_element.button
        self.assertEqual(c.use_container_width, False)