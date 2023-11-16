"""Unit tests for the generate python code prompt class"""
import sys
import pandas as pd
import pytest
from pandasai import SmartDataframe
from pandasai.helpers.output_types import output_type_factory, DefaultOutputType, output_types_map
from pandasai.prompts import GeneratePythonCodePrompt
from pandasai.llm.fake import FakeLLM
from pandasai.helpers.viz_library_types import MatplotlibVizLibraryType, viz_lib_map, viz_lib_type_factory

class TestGeneratePythonCodePrompt:
    """Unit tests for the generate python code prompt class"""

    @pytest.mark.parametrize('save_charts_path,output_type_hint,viz_library_type_hint', [('exports/charts', DefaultOutputType().template_hint, MatplotlibVizLibraryType().template_hint), ('custom/dir/for/charts', DefaultOutputType().template_hint, MatplotlibVizLibraryType().template_hint), *[('exports/charts', output_type_factory(type_).template_hint, viz_lib_type_factory(viz_type_).template_hint) for type_ in output_types_map for viz_type_ in viz_lib_map]])
    def test_str_with_args(self, save_charts_path, output_type_hint, viz_library_type_hint):
        if False:
            print('Hello World!')
        'Test casting of prompt to string and interpolation of context.\n\n        Parameterized for the following cases:\n        * `save_charts_path` is "exports/charts", `output_type_hint` is default,\n        `viz_library_type_hint` is default\n        * `save_charts_path` is "custom/dir/for/charts", `output_type_hint`\n            is default, `viz_library_type_hint` is default\n        * `save_charts_path` is "exports/charts", `output_type_hint` any of\n            possible types in `pandasai.helpers.output_types.output_types_map`,\n            `viz_library_type_hint` any of\n            possible types in `pandasai.helpers.viz_library_types.viz_library_types_map`\n        '
        llm = FakeLLM('plt.show()')
        dfs = [SmartDataframe(pd.DataFrame({'a': [1], 'b': [4]}), config={'llm': llm})]
        prompt = GeneratePythonCodePrompt()
        prompt.set_var('dfs', dfs)
        prompt.set_var('conversation', 'Question')
        prompt.set_var('save_charts_path', save_charts_path)
        prompt.set_var('output_type_hint', output_type_hint)
        prompt.set_var('viz_library_type', viz_library_type_hint)
        prompt.set_var('skills', '')
        expected_prompt_content = f'You are provided with the following pandas DataFrames:\n\n<dataframe>\nDataframe dfs[0], with 1 rows and 2 columns.\nThis is the metadata of the dataframe dfs[0]:\na,b\n1,4\n</dataframe>\n\n<conversation>\nQuestion\n</conversation>\n\nThis is the initial python function. Do not change the params. Given the context, use the right dataframes.\n```python\n# TODO import all the dependencies required\nimport pandas as pd\n\ndef analyze_data(dfs: list[pd.DataFrame]) -> dict:\n    """\n    Analyze the data, using the provided dataframes (`dfs`).\n    1. Prepare: Preprocessing and cleaning data if necessary\n    2. Process: Manipulating data for analysis (grouping, filtering, aggregating, etc.)\n    3. Analyze: Conducting the actual analysis (if the user asks to plot a chart you must save it as an image in temp_chart.png and not show the chart.)\n    {viz_library_type_hint}\n    At the end, return a dictionary of:\n    {output_type_hint}\n    """\n```\n\nTake a deep breath and reason step-by-step. Act as a senior data analyst.\nIn the answer, you must never write the "technical" names of the tables.\nBased on the last message in the conversation:\n- return the updated analyze_data function wrapped within ```python ```'
        actual_prompt_content = prompt.to_string()
        if sys.platform.startswith('win'):
            actual_prompt_content = actual_prompt_content.replace('\r\n', '\n')
        assert actual_prompt_content == expected_prompt_content

    def test_advanced_reasoning_prompt(self):
        if False:
            return 10
        '\n        Test a prompt with advanced reasoning framework\n        '
        llm = FakeLLM('plt.show()')
        viz_library_type_hint = ''
        dfs = [SmartDataframe(pd.DataFrame({'a': [1], 'b': [4]}), config={'llm': llm, 'use_advanced_reasoning_framework': True})]
        prompt = GeneratePythonCodePrompt()
        prompt.set_config(dfs[0]._lake.config)
        prompt.set_var('dfs', dfs)
        prompt.set_var('conversation', 'Question')
        prompt.set_var('output_type_hint', '')
        prompt.set_var('skills', '')
        prompt.set_var('viz_library_type', '')
        expected_prompt_content = f'You are provided with the following pandas DataFrames:\n\n<dataframe>\nDataframe dfs[0], with 1 rows and 2 columns.\nThis is the metadata of the dataframe dfs[0]:\na,b\n1,4\n</dataframe>\n\n<conversation>\nQuestion\n</conversation>\n\nThis is the initial python function. Do not change the params. Given the context, use the right dataframes.\n```python\n# TODO import all the dependencies required\nimport pandas as pd\n\ndef analyze_data(dfs: list[pd.DataFrame]) -> dict:\n    """\n    Analyze the data, using the provided dataframes (`dfs`).\n    1. Prepare: Preprocessing and cleaning data if necessary\n    2. Process: Manipulating data for analysis (grouping, filtering, aggregating, etc.)\n    3. Analyze: Conducting the actual analysis (if the user asks to plot a chart you must save it as an image in temp_chart.png and not show the chart.)\n    {viz_library_type_hint}\n    At the end, return a dictionary of:\n    \n    """\n```\n\nTake a deep breath and reason step-by-step. Act as a senior data analyst.\nIn the answer, you must never write the "technical" names of the tables.\nBased on the last message in the conversation:\n- explain your reasoning to implement the last step to the user that asked for it; it should be wrapped between <reasoning> tags.\n- answer to the user as you would do as a data analyst; wrap it between <answer> tags; do not include the value or the chart itself (it will be calculated later).\n- return the updated analyze_data function wrapped within ```python ```'
        actual_prompt_content = prompt.to_string()
        print(expected_prompt_content)
        if sys.platform.startswith('win'):
            actual_prompt_content = actual_prompt_content.replace('\r\n', '\n')
        assert actual_prompt_content == expected_prompt_content

    def test_custom_instructions(self):
        if False:
            print('Hello World!')
        custom_instructions = 'Analyze the data.\n1. Load: Load the data from a file or database\n2. Prepare: Preprocessing and cleaning data if necessary\n3. Process: Manipulating data for analysis (grouping, filtering, aggregating, etc.)\n4. Analyze: Conducting the actual analysis (if the user asks to plot a chart you must save it as an image in temp_chart.png and not show the chart.)'
        prompt = GeneratePythonCodePrompt(custom_instructions=custom_instructions)
        actual_instructions = prompt._args['instructions']
        assert actual_instructions == 'Analyze the data.\n    1. Load: Load the data from a file or database\n    2. Prepare: Preprocessing and cleaning data if necessary\n    3. Process: Manipulating data for analysis (grouping, filtering, aggregating, etc.)\n    4. Analyze: Conducting the actual analysis (if the user asks to plot a chart you must save it as an image in temp_chart.png and not show the chart.)'