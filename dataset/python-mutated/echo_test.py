from parameterized import parameterized
import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class EchoTest(DeltaGeneratorTestCase):

    @parameterized.expand([('code_location default', lambda : st.echo(), 0, 1), ('code_location above', lambda : st.echo('above'), 0, 1), ('code_location below', lambda : st.echo('below'), 1, 0)])
    def test_echo(self, _, echo, echo_index, output_index):
        if False:
            for i in range(10):
                print('nop')
        with echo():
            st.write('Hello')
            'hi'

            def foo(x):
                if False:
                    while True:
                        i = 10
                y = x + 10
                print(y)

            class MyClass(object):

                def do_x(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    pass

                def do_y(self):
                    if False:
                        return 10
                    pass
        echo_str = 'st.write("Hello")\n\n"hi"\n\ndef foo(x):\n    y = x + 10\n\n    print(y)\n\nclass MyClass(object):\n    def do_x(self):\n        pass\n\n    def do_y(self):\n        pass'
        element = self.get_delta_from_queue(echo_index).new_element
        self.assertEqual(echo_str, element.code.code_text)
        element = self.get_delta_from_queue(output_index).new_element
        self.assertEqual('Hello', element.markdown.body)
        self.clear_queue()

    @parameterized.expand([('code_location default', {}, 0, 1), ('code_location above', {'code_location': 'above'}, 0, 1), ('code_location below', {'code_location': 'below'}, 1, 0)])
    def test_echo_unindent(self, _, echo_kwargs_very_long_name_very_long_very_very_very_very_very_very_long, echo_index, output_index):
        if False:
            return 10
        with st.echo(**echo_kwargs_very_long_name_very_long_very_very_very_very_very_very_long):
            st.write('Hello')
            'hi'

            def foo(x):
                if False:
                    print('Hello World!')
                y = x + 10
                print(y)

            class MyClass(object):

                def do_x(self):
                    if False:
                        print('Hello World!')
                    pass

                def do_y(self):
                    if False:
                        return 10
                    pass
        echo_str = 'st.write("Hello")\n"hi"\n\ndef foo(x):\n    y = x + 10\n\n    print(y)\n\nclass MyClass(object):\n    def do_x(self):\n        pass\n\n    def do_y(self):\n        pass'
        element = self.get_delta_from_queue(echo_index).new_element
        self.assertEqual(echo_str, element.code.code_text)
        element = self.get_delta_from_queue(output_index).new_element
        self.assertEqual('Hello', element.markdown.body)
        self.clear_queue()

    def test_if_elif_else(self):
        if False:
            return 10
        page = 'Dual'
        if page == 'Single':
            with st.echo():
                st.write('Single')
        elif page == 'Dual':
            with st.echo():
                st.write('Dual')
        else:
            with st.echo():
                st.write('ELSE')
        echo_str = 'st.write("Dual")'
        element = self.get_delta_from_queue(0).new_element
        self.assertEqual(echo_str, element.code.code_text)
        element = self.get_delta_from_queue(1).new_element
        self.assertEqual('Dual', element.markdown.body)
        self.clear_queue()

    def test_root_level_echo(self):
        if False:
            return 10
        import tests.streamlit.echo_test_data.root_level_echo
        echo_str = 'a = 123'
        element = self.get_delta_from_queue(0).new_element
        self.assertEqual(echo_str, element.code.code_text)

    def test_echo_multiline_param(self):
        if False:
            return 10
        import tests.streamlit.echo_test_data.multiline_param_echo
        echo_str = 'a = 123'
        element = self.get_delta_from_queue(0).new_element
        self.assertEqual(echo_str, element.code.code_text)