"""Balloons unit test."""
import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class BallonsTest(DeltaGeneratorTestCase):
    """Test ability to marshall balloons protos."""

    def test_st_balloons(self):
        if False:
            while True:
                i = 10
        'Test st.balloons.'
        st.balloons()
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.balloons.show, True)