import streamlit as st
from streamlit.proto.Markdown_pb2 import Markdown as MarkdownProto
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class CodeElement(DeltaGeneratorTestCase):
    """Test ability to marshall code protos."""

    def test_st_code_default(self):
        if False:
            for i in range(10):
                print('nop')
        'Test st.code() with default language (python).'
        code = "print('Hello, %s!' % 'Streamlit')"
        st.code(code)
        element = self.get_delta_from_queue().new_element
        self.assertEqual(element.code.code_text, code)
        self.assertEqual(element.code.show_line_numbers, False)
        self.assertEqual(element.code.language, 'python')

    def test_st_code_python(self):
        if False:
            while True:
                i = 10
        'Test st.code with python language.'
        code = "print('My string = %d' % my_value)"
        st.code(code, language='python')
        element = self.get_delta_from_queue().new_element
        self.assertEqual(element.code.code_text, code)
        self.assertEqual(element.code.show_line_numbers, False)
        self.assertEqual(element.code.language, 'python')

    def test_st_code_none(self):
        if False:
            i = 10
            return i + 15
        'Test st.code with None language.'
        code = "print('My string = %d' % my_value)"
        st.code(code, language=None)
        element = self.get_delta_from_queue().new_element
        self.assertEqual(element.code.code_text, code)
        self.assertEqual(element.code.show_line_numbers, False)
        self.assertEqual(element.code.language, 'plaintext')

    def test_st_code_none_with_line_numbers(self):
        if False:
            return 10
        'Test st.code with None language and line numbers.'
        code = "print('My string = %d' % my_value)"
        st.code(code, language=None, line_numbers=True)
        element = self.get_delta_from_queue().new_element
        self.assertEqual(element.code.code_text, code)
        self.assertEqual(element.code.show_line_numbers, True)
        self.assertEqual(element.code.language, 'plaintext')

    def test_st_code_python_with_line_numbers(self):
        if False:
            while True:
                i = 10
        'Test st.code with Python language and line numbers.'
        code = "print('My string = %d' % my_value)"
        st.code(code, language='python', line_numbers=True)
        element = self.get_delta_from_queue().new_element
        self.assertEqual(element.code.code_text, code)
        self.assertEqual(element.code.show_line_numbers, True)
        self.assertEqual(element.code.language, 'python')