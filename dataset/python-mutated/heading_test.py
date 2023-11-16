import pytest
import streamlit as st
from streamlit.errors import StreamlitAPIException
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class StHeaderTest(DeltaGeneratorTestCase):
    """Test ability to marshall header protos."""

    def test_st_header(self):
        if False:
            while True:
                i = 10
        'Test st.header.'
        st.header('some header')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some header')
        self.assertEqual(el.heading.tag, 'h2')
        self.assertFalse(el.heading.hide_anchor, False)
        self.assertFalse(el.heading.divider)

    def test_st_header_with_anchor(self):
        if False:
            for i in range(10):
                print('nop')
        'Test st.header with anchor.'
        st.header('some header', anchor='some-anchor')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some header')
        self.assertEqual(el.heading.tag, 'h2')
        self.assertEqual(el.heading.anchor, 'some-anchor')
        self.assertFalse(el.heading.hide_anchor, False)
        self.assertFalse(el.heading.divider)

    def test_st_header_with_hidden_anchor(self):
        if False:
            return 10
        'Test st.header with hidden anchor.'
        st.header('some header', anchor=False)
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some header')
        self.assertEqual(el.heading.tag, 'h2')
        self.assertEqual(el.heading.anchor, '')
        self.assertTrue(el.heading.hide_anchor, True)
        self.assertFalse(el.heading.divider)

    def test_st_header_with_invalid_anchor(self):
        if False:
            while True:
                i = 10
        'Test st.header with invalid anchor.'
        with pytest.raises(StreamlitAPIException):
            st.header('some header', anchor=True)

    def test_st_header_with_help(self):
        if False:
            print('Hello World!')
        'Test st.header with help.'
        st.header('some header', help='help text')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some header')
        self.assertEqual(el.heading.tag, 'h2')
        self.assertEqual(el.heading.help, 'help text')
        self.assertFalse(el.heading.divider)

    def test_st_header_with_divider_true(self):
        if False:
            while True:
                i = 10
        'Test st.header with divider True.'
        st.header('some header', divider=True)
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some header')
        self.assertEqual(el.heading.tag, 'h2')
        self.assertFalse(el.heading.hide_anchor, False)
        self.assertEqual(el.heading.divider, 'auto')

    def test_st_header_with_divider_color(self):
        if False:
            print('Hello World!')
        'Test st.header with divider color.'
        st.header('some header', divider='blue')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some header')
        self.assertEqual(el.heading.tag, 'h2')
        self.assertFalse(el.heading.hide_anchor, False)
        self.assertEqual(el.heading.divider, 'blue')

    def test_st_header_with_invalid_divider(self):
        if False:
            while True:
                i = 10
        'Test st.header with invalid divider.'
        with pytest.raises(StreamlitAPIException):
            st.header('some header', divider='corgi')

class StSubheaderTest(DeltaGeneratorTestCase):
    """Test ability to marshall subheader protos."""

    def test_st_subheader(self):
        if False:
            print('Hello World!')
        'Test st.subheader.'
        st.subheader('some subheader')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some subheader')
        self.assertEqual(el.heading.tag, 'h3')
        self.assertFalse(el.heading.hide_anchor)
        self.assertFalse(el.heading.divider)

    def test_st_subheader_with_anchor(self):
        if False:
            return 10
        'Test st.subheader with anchor.'
        st.subheader('some subheader', anchor='some-anchor')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some subheader')
        self.assertEqual(el.heading.tag, 'h3')
        self.assertEqual(el.heading.anchor, 'some-anchor')
        self.assertFalse(el.heading.hide_anchor)
        self.assertFalse(el.heading.divider)

    def test_st_subheader_with_hidden_anchor(self):
        if False:
            i = 10
            return i + 15
        'Test st.subheader with hidden anchor.'
        st.subheader('some subheader', anchor=False)
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some subheader')
        self.assertEqual(el.heading.tag, 'h3')
        self.assertEqual(el.heading.anchor, '')
        self.assertTrue(el.heading.hide_anchor, True)
        self.assertFalse(el.heading.divider)

    def test_st_subheader_with_invalid_anchor(self):
        if False:
            print('Hello World!')
        'Test st.subheader with invalid anchor.'
        with pytest.raises(StreamlitAPIException):
            st.subheader('some header', anchor=True)

    def test_st_subheader_with_help(self):
        if False:
            while True:
                i = 10
        'Test st.subheader with help.'
        st.subheader('some subheader', help='help text')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some subheader')
        self.assertEqual(el.heading.tag, 'h3')
        self.assertEqual(el.heading.help, 'help text')
        self.assertFalse(el.heading.divider)

    def test_st_subheader_with_divider_true(self):
        if False:
            return 10
        'Test st.subheader with divider True.'
        st.subheader('some subheader', divider=True)
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some subheader')
        self.assertEqual(el.heading.tag, 'h3')
        self.assertFalse(el.heading.hide_anchor)
        self.assertEqual(el.heading.divider, 'auto')

    def test_st_subheader_with_divider_color(self):
        if False:
            for i in range(10):
                print('nop')
        'Test st.subheader with divider color.'
        st.subheader('some subheader', divider='blue')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some subheader')
        self.assertEqual(el.heading.tag, 'h3')
        self.assertFalse(el.heading.hide_anchor)
        self.assertEqual(el.heading.divider, 'blue')

    def test_st_subheader_with_invalid_divider(self):
        if False:
            while True:
                i = 10
        'Test st.subheader with invalid divider.'
        with pytest.raises(StreamlitAPIException):
            st.subheader('some header', divider='corgi')

class StTitleTest(DeltaGeneratorTestCase):
    """Test ability to marshall title protos."""

    def test_st_title(self):
        if False:
            i = 10
            return i + 15
        'Test st.title.'
        st.title('some title')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some title')
        self.assertEqual(el.heading.tag, 'h1')
        self.assertFalse(el.heading.hide_anchor)
        self.assertFalse(el.heading.divider)

    def test_st_title_with_anchor(self):
        if False:
            i = 10
            return i + 15
        'Test st.title with anchor.'
        st.title('some title', anchor='some-anchor')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some title')
        self.assertEqual(el.heading.tag, 'h1')
        self.assertEqual(el.heading.anchor, 'some-anchor')
        self.assertFalse(el.heading.hide_anchor)
        self.assertFalse(el.heading.divider)

    def test_st_title_with_hidden_anchor(self):
        if False:
            return 10
        'Test st.title with hidden anchor.'
        st.title('some title', anchor=False)
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some title')
        self.assertEqual(el.heading.tag, 'h1')
        self.assertEqual(el.heading.anchor, '')
        self.assertTrue(el.heading.hide_anchor)
        self.assertFalse(el.heading.divider)

    def test_st_title_with_invalid_anchor(self):
        if False:
            print('Hello World!')
        'Test st.title with invalid anchor.'
        with pytest.raises(StreamlitAPIException, match='Anchor parameter has invalid value:'):
            st.title('some header', anchor=True)
        with pytest.raises(StreamlitAPIException, match='Anchor parameter has invalid type:'):
            st.title('some header', anchor=6)

    def test_st_title_with_help(self):
        if False:
            i = 10
            return i + 15
        'Test st.title with help.'
        st.title('some title', help='help text')
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.heading.body, 'some title')
        self.assertEqual(el.heading.tag, 'h1')
        self.assertEqual(el.heading.help, 'help text')
        self.assertFalse(el.heading.divider)

    def test_st_title_with_invalid_divider(self):
        if False:
            while True:
                i = 10
        'Test st.title with invalid divider.'
        with pytest.raises(TypeError):
            st.title('some header', divider=True)
        with pytest.raises(TypeError):
            st.title('some header', divider='blue')