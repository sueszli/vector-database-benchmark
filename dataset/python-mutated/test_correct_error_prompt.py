"""Unit tests for the correct error prompt class"""
import sys
import pandas as pd
from pandasai import SmartDataframe
from pandasai.prompts import CorrectErrorPrompt
from pandasai.llm.fake import FakeLLM

class TestCorrectErrorPrompt:
    """Unit tests for the correct error prompt class"""

    def test_str_with_args(self):
        if False:
            i = 10
            return i + 15
        'Test that the __str__ method is implemented'
        llm = FakeLLM('plt.show()')
        dfs = [SmartDataframe(pd.DataFrame({}), config={'llm': llm})]
        prompt = CorrectErrorPrompt(engine='pandas', code='df.head()', error_returned='Error message')
        prompt.set_var('dfs', dfs)
        prompt.set_var('conversation', 'What is the correct code?')
        prompt_content = prompt.to_string()
        if sys.platform.startswith('win'):
            prompt_content = prompt_content.replace('\r\n', '\n')
        assert prompt_content == '\nYou are provided with the following pandas DataFrames with the following metadata:\n\n<dataframe>\nDataframe dfs[0], with 0 rows and 0 columns.\nThis is the metadata of the dataframe dfs[0]:\n\n</dataframe>\n\nThe user asked the following question:\nWhat is the correct code?\n\nYou generated this python code:\ndf.head()\n\nIt fails with the following error:\nError message\n\nCorrect the python code and return a new python code that fixes the above mentioned error. Do not generate the same code again.\n'