import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class StMarkdownAPITest(DeltaGeneratorTestCase):
    """Test st.markdown API."""

    def test_st_markdown(self):
        if False:
            i = 10
            return i + 15
        'Test st.markdown.'
        st.markdown('    some markdown  ')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.markdown.body, 'some markdown')
        st.markdown('    some markdown  ', unsafe_allow_html=True)
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.markdown.body, 'some markdown')
        self.assertTrue(el.markdown.allow_html)
        st.markdown('    some markdown  ', help='help text')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.markdown.body, 'some markdown')
        self.assertEqual(el.markdown.help, 'help text')

class StCaptionAPITest(DeltaGeneratorTestCase):
    """Test st.caption APIs."""

    def test_st_caption_with_help(self):
        if False:
            while True:
                i = 10
        'Test st.caption with help.'
        st.caption('some caption', help='help text')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.markdown.help, 'help text')

class StLatexAPITest(DeltaGeneratorTestCase):
    """Test st.latex APIs."""

    def test_st_latex_with_help(self):
        if False:
            print('Hello World!')
        'Test st.latex with help.'
        st.latex('\n            a + ar + a r^2 + a r^3 + \\cdots + a r^{n-1} =\n            \\sum_{k=0}^{n-1} ar^k =\n            a \\left(\\frac{1-r^{n}}{1-r}\\right)\n            ', help='help text')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.markdown.help, 'help text')