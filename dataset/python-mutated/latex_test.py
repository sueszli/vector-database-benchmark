"""LaTeX unit test."""
import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class LatexTest(DeltaGeneratorTestCase):
    """Test ability to marshall latex protos."""

    def test_latex(self):
        if False:
            while True:
                i = 10
        st.latex('ax^2 + bx + c = 0')
        c = self.get_delta_from_queue().new_element.markdown
        self.assertEqual(c.body, '$$\nax^2 + bx + c = 0\n$$')

    def test_sympy_expression(self):
        if False:
            while True:
                i = 10
        try:
            import sympy
            (a, b) = sympy.symbols('a b')
            out = a + b
        except:
            out = 'a + b'
        st.latex(out)
        c = self.get_delta_from_queue().new_element.markdown
        self.assertEqual(c.body, '$$\na + b\n$$')