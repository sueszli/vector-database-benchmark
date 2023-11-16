import os
from unittest import mock
import streamlit as st
from tests.delta_generator_test_case import DeltaGeneratorTestCase

def patch_varname_getter():
    if False:
        return 10
    '\n    Patches streamlit.elements.doc_string so _get_variable_name()\n    works outside ScriptRunner.\n    '
    import inspect
    parent_frame_filename = inspect.getouterframes(inspect.currentframe())[2].filename
    return mock.patch('streamlit.elements.doc_string.SCRIPTRUNNER_FILENAME', parent_frame_filename)

class StHelpAPITest(DeltaGeneratorTestCase):
    """Test Public Streamlit Public APIs."""

    def test_st_help(self):
        if False:
            print('Hello World!')
        'Test st.help.'
        with patch_varname_getter():
            st.help(os.chdir)
        el = self.get_delta_from_queue().new_element.doc_string
        self.assertEqual('os.chdir', el.name)
        self.assertEqual('builtin_function_or_method', el.type)
        self.assertTrue(el.doc_string.startswith('Change the current working directory'))
        self.assertEqual(f'posix.chdir(path)', el.value)