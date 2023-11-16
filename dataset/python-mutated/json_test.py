import numpy as np
import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class StJsonAPITest(DeltaGeneratorTestCase):
    """Test Public Streamlit Public APIs."""

    def test_st_json(self):
        if False:
            while True:
                i = 10
        'Test st.json.'
        st.json('{"some": "json"}')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.json.body, '{"some": "json"}')
        n = np.array([1, 2, 3, 4, 5])
        data = {n[0]: 'this key will not render as JSON', 'array': n}
        st.json(data)
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.json.body, '{"array": "array([1, 2, 3, 4, 5])"}')