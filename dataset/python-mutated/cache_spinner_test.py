"""Unit tests for cache's show_spinner option."""
from unittest.mock import Mock, patch
import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase

@st.cache(show_spinner=False)
def function_without_spinner():
    if False:
        for i in range(10):
            print('nop')
    return 3

@st.cache(show_spinner=True)
def function_with_spinner():
    if False:
        while True:
            i = 10
    return 3

class CacheSpinnerTest(DeltaGeneratorTestCase):
    """
    We test the ability to turn on and off the spinner with the show_spinner
    option by inspecting the report queue.
    """

    @patch('streamlit.runtime.legacy_caching.caching.show_deprecation_warning', Mock())
    def test_with_spinner(self):
        if False:
            return 10
        'If the show_spinner flag is set, there should be one element in the\n        report queue.\n        '
        function_with_spinner()
        self.assertFalse(self.forward_msg_queue.is_empty())

    @patch('streamlit.runtime.legacy_caching.caching.show_deprecation_warning', Mock())
    def test_without_spinner(self):
        if False:
            return 10
        'If the show_spinner flag is not set, the report queue should be\n        empty.\n        '
        function_without_spinner()
        self.assertTrue(self.forward_msg_queue.is_empty())