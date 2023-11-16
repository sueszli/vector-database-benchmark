# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from parameterized import parameterized

import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase


class EchoTest(DeltaGeneratorTestCase):
    @parameterized.expand(
        [
            ("code_location default", lambda: st.echo(), 0, 1),
            ("code_location above", lambda: st.echo("above"), 0, 1),
            ("code_location below", lambda: st.echo("below"), 1, 0),
        ]
    )
    def test_echo(self, _, echo, echo_index, output_index):
        # The empty lines below are part of the test. Do not remove them.
        with echo():
            st.write("Hello")

            "hi"

            def foo(x):
                y = x + 10

                print(y)

            class MyClass(object):
                def do_x(self):
                    pass

                def do_y(self):
                    pass

        echo_str = """st.write("Hello")

"hi"

def foo(x):
    y = x + 10

    print(y)

class MyClass(object):
    def do_x(self):
        pass

    def do_y(self):
        pass"""

        element = self.get_delta_from_queue(echo_index).new_element
        self.assertEqual(echo_str, element.code.code_text)

        element = self.get_delta_from_queue(output_index).new_element
        self.assertEqual("Hello", element.markdown.body)

        self.clear_queue()

    @parameterized.expand(
        [
            ("code_location default", {}, 0, 1),
            ("code_location above", {"code_location": "above"}, 0, 1),
            ("code_location below", {"code_location": "below"}, 1, 0),
        ]
    )
    def test_echo_unindent(
        self,
        _,
        echo_kwargs_very_long_name_very_long_very_very_very_very_very_very_long,
        echo_index,
        output_index,
    ):
        with st.echo(
            **echo_kwargs_very_long_name_very_long_very_very_very_very_very_very_long
        ):
            st.write("Hello")
            "hi"

            def foo(x):
                y = x + 10

                print(y)

            class MyClass(object):
                def do_x(self):
                    pass

                def do_y(self):
                    pass

        echo_str = """st.write("Hello")
"hi"

def foo(x):
    y = x + 10

    print(y)

class MyClass(object):
    def do_x(self):
        pass

    def do_y(self):
        pass"""

        element = self.get_delta_from_queue(echo_index).new_element
        self.assertEqual(echo_str, element.code.code_text)
        element = self.get_delta_from_queue(output_index).new_element
        self.assertEqual("Hello", element.markdown.body)
        self.clear_queue()

    def test_if_elif_else(self):
        page = "Dual"

        if page == "Single":
            with st.echo():
                st.write("Single")

        elif page == "Dual":
            with st.echo():
                st.write("Dual")

        else:
            with st.echo():
                st.write("ELSE")

        echo_str = 'st.write("Dual")'
        element = self.get_delta_from_queue(0).new_element
        self.assertEqual(echo_str, element.code.code_text)
        element = self.get_delta_from_queue(1).new_element
        self.assertEqual("Dual", element.markdown.body)
        self.clear_queue()

    def test_root_level_echo(self):
        import tests.streamlit.echo_test_data.root_level_echo

        echo_str = "a = 123"

        element = self.get_delta_from_queue(0).new_element
        self.assertEqual(echo_str, element.code.code_text)

    def test_echo_multiline_param(self):
        import tests.streamlit.echo_test_data.multiline_param_echo

        echo_str = "a = 123"

        element = self.get_delta_from_queue(0).new_element
        self.assertEqual(echo_str, element.code.code_text)
