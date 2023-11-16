"""Divider unit test."""
import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class DividerTest(DeltaGeneratorTestCase):
    """Test ability to marshall divider protos."""

    def test_divider(self):
        if False:
            print('Hello World!')
        st.divider()
        c = self.get_delta_from_queue().new_element.markdown
        self.assertEqual(c.body, '---')