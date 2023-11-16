import unittest
from dataclasses import FrozenInstanceError
import pytest
from streamlit.runtime.script_data import ScriptData

class ScriptDataTest(unittest.TestCase):

    def test_script_folder_and_name_set(self):
        if False:
            print('Hello World!')
        script_data = ScriptData('/path/to/some/script.py', 'streamlit run /path/to/some/script.py')
        assert script_data.main_script_path == '/path/to/some/script.py'
        assert script_data.command_line == 'streamlit run /path/to/some/script.py'
        assert script_data.script_folder == '/path/to/some'
        assert script_data.name == 'script'

    def test_is_frozen(self):
        if False:
            i = 10
            return i + 15
        script_data = ScriptData('/path/to/some/script.py', 'streamlit run /path/to/some/script.py')
        with pytest.raises(FrozenInstanceError):
            script_data.name = 'bob'