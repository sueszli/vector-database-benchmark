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

"""selectbox unit tests."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from parameterized import parameterized

import streamlit as st
from streamlit.errors import StreamlitAPIException
from streamlit.proto.LabelVisibilityMessage_pb2 import LabelVisibilityMessage
from streamlit.testing.v1.app_test import AppTest
from streamlit.testing.v1.util import patch_config_options
from tests.delta_generator_test_case import DeltaGeneratorTestCase


class SelectboxTest(DeltaGeneratorTestCase):
    """Test ability to marshall selectbox protos."""

    def test_just_label(self):
        """Test that it can be called with no value."""
        st.selectbox("the label", ("m", "f"))

        c = self.get_delta_from_queue().new_element.selectbox
        self.assertEqual(c.label, "the label")
        self.assertEqual(
            c.label_visibility.value,
            LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE,
        )
        self.assertEqual(c.default, 0)
        self.assertEqual(c.HasField("default"), True)
        self.assertEqual(c.disabled, False)
        self.assertEqual(c.placeholder, "Choose an option")

    def test_just_disabled(self):
        """Test that it can be called with disabled param."""
        st.selectbox("the label", ("m", "f"), disabled=True)

        c = self.get_delta_from_queue().new_element.selectbox
        self.assertEqual(c.disabled, True)

    def test_valid_value(self):
        """Test that valid value is an int."""
        st.selectbox("the label", ("m", "f"), 1)

        c = self.get_delta_from_queue().new_element.selectbox
        self.assertEqual(c.label, "the label")
        self.assertEqual(c.default, 1)

    def test_none_index(self):
        """Test that it can be called with None as index value."""
        st.selectbox("the label", ("m", "f"), index=None)

        c = self.get_delta_from_queue().new_element.selectbox
        self.assertEqual(c.label, "the label")
        # If a proto property is null is not determined by this value,
        # but by the check via the HasField method:
        self.assertEqual(c.default, 0)
        self.assertEqual(c.HasField("default"), False)

    def test_noneType_option(self):
        """Test NoneType option value."""
        current_value = st.selectbox("the label", (None, "selected"), 0)

        self.assertEqual(current_value, None)

    @parameterized.expand(
        [
            (("m", "f"), ["m", "f"]),
            (["male", "female"], ["male", "female"]),
            (np.array(["m", "f"]), ["m", "f"]),
            (pd.Series(np.array(["male", "female"])), ["male", "female"]),
            (pd.DataFrame({"options": ["male", "female"]}), ["male", "female"]),
            (
                pd.DataFrame(
                    data=[[1, 4, 7], [2, 5, 8], [3, 6, 9]], columns=["a", "b", "c"]
                ).columns,
                ["a", "b", "c"],
            ),
        ]
    )
    def test_option_types(self, options, proto_options):
        """Test that it supports different types of options."""
        st.selectbox("the label", options)

        c = self.get_delta_from_queue().new_element.selectbox
        self.assertEqual(c.label, "the label")
        self.assertEqual(c.default, 0)
        self.assertEqual(c.options, proto_options)

    def test_not_iterable_option_types(self):
        """Test that it supports different types of options."""
        with pytest.raises(TypeError):
            st.selectbox("the label", 123)

    def test_cast_options_to_string(self):
        """Test that it casts options to string."""
        arg_options = ["some str", 123, None, {}]
        proto_options = ["some str", "123", "None", "{}"]

        st.selectbox("the label", arg_options)

        c = self.get_delta_from_queue().new_element.selectbox
        self.assertEqual(c.label, "the label")
        self.assertEqual(c.default, 0)
        self.assertEqual(c.options, proto_options)

    def test_format_function(self):
        """Test that it formats options."""
        arg_options = [{"name": "john", "height": 180}, {"name": "lisa", "height": 200}]
        proto_options = ["john", "lisa"]

        st.selectbox("the label", arg_options, format_func=lambda x: x["name"])

        c = self.get_delta_from_queue().new_element.selectbox
        self.assertEqual(c.label, "the label")
        self.assertEqual(c.default, 0)
        self.assertEqual(c.options, proto_options)

    @parameterized.expand([((),), ([],), (np.array([]),), (pd.Series(np.array([])),)])
    def test_no_options(self, options):
        """Test that it handles no options."""
        st.selectbox("the label", options)

        c = self.get_delta_from_queue().new_element.selectbox
        self.assertEqual(c.label, "the label")
        self.assertEqual(c.default, 0)
        self.assertEqual(c.options, [])

    def test_invalid_value(self):
        """Test that value must be an int."""
        with self.assertRaises(StreamlitAPIException):
            st.selectbox("the label", ("m", "f"), "1")

    def test_invalid_value_range(self):
        """Test that value must be within the length of the options."""
        with self.assertRaises(StreamlitAPIException):
            st.selectbox("the label", ("m", "f"), 2)

    def test_outside_form(self):
        """Test that form id is marshalled correctly outside of a form."""

        st.selectbox("foo", ("bar", "baz"))

        proto = self.get_delta_from_queue().new_element.color_picker
        self.assertEqual(proto.form_id, "")

    @patch("streamlit.runtime.Runtime.exists", MagicMock(return_value=True))
    def test_inside_form(self):
        """Test that form id is marshalled correctly inside of a form."""

        with st.form("form"):
            st.selectbox("foo", ("bar", "baz"))

        # 2 elements will be created: form block, widget
        self.assertEqual(len(self.get_all_deltas_from_queue()), 2)

        form_proto = self.get_delta_from_queue(0).add_block
        selectbox_proto = self.get_delta_from_queue(1).new_element.selectbox
        self.assertEqual(selectbox_proto.form_id, form_proto.form.form_id)

    @parameterized.expand(
        [
            ("visible", LabelVisibilityMessage.LabelVisibilityOptions.VISIBLE),
            ("hidden", LabelVisibilityMessage.LabelVisibilityOptions.HIDDEN),
            ("collapsed", LabelVisibilityMessage.LabelVisibilityOptions.COLLAPSED),
        ]
    )
    def test_label_visibility(self, label_visibility_value, proto_value):
        """Test that it can be called with label_visibility param."""
        st.selectbox("the label", ("m", "f"), label_visibility=label_visibility_value)

        c = self.get_delta_from_queue().new_element.selectbox
        self.assertEqual(c.label_visibility.value, proto_value)

    def test_label_visibility_wrong_value(self):
        with self.assertRaises(StreamlitAPIException) as e:
            st.selectbox("the label", ("m", "f"), label_visibility="wrong_value")
        self.assertEqual(
            str(e.exception),
            "Unsupported label_visibility option 'wrong_value'. Valid values are "
            "'visible', 'hidden' or 'collapsed'.",
        )

    def test_placeholder(self):
        """Test that it can be called with placeholder params."""
        st.selectbox("the label", ("m", "f"), placeholder="Please select")

        c = self.get_delta_from_queue().new_element.selectbox
        self.assertEqual(c.placeholder, "Please select")


def test_selectbox_interaction():
    """Test interactions with an empty selectbox widget."""

    def script():
        import streamlit as st

        st.selectbox("the label", ("m", "f"), index=None)

    at = AppTest.from_function(script).run()
    selectbox = at.selectbox[0]
    assert selectbox.value is None

    # Select option m
    at = selectbox.set_value("m").run()
    selectbox = at.selectbox[0]
    assert selectbox.value == "m"

    # # Clear the value
    at = selectbox.set_value(None).run()
    selectbox = at.selectbox[0]
    assert selectbox.value is None


def test_selectbox_enum_coercion():
    """Test E2E Enum Coercion on a selectbox."""

    def script():
        from enum import Enum

        import streamlit as st

        class EnumA(Enum):
            A = 1
            B = 2
            C = 3

        selected = st.selectbox("my_enum", EnumA, index=0)
        st.text(id(selected.__class__))
        st.text(id(EnumA))
        st.text(selected in EnumA)

    at = AppTest.from_function(script).run()

    def test_enum():
        selectbox = at.selectbox[0]
        original_class = selectbox.value.__class__
        selectbox.set_value(original_class.C).run()
        assert at.text[0].value == at.text[1].value, "Enum Class ID not the same"
        assert at.text[2].value == "True", "Not all enums found in class"

    with patch_config_options({"runner.enumCoercion": "nameOnly"}):
        test_enum()
    with patch_config_options({"runner.enumCoercion": "off"}):
        with pytest.raises(AssertionError):
            test_enum()  # expect a failure with the config value off.
