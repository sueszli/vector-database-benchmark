import unittest
from typing import Optional, Union, List, Dict, Any
from unittest import mock
import pytest
from haystack import Pipeline, Answer, Document, BaseComponent, MultiLabel
from haystack.agents.base import ToolsManager, Tool

@pytest.fixture
def tools_manager():
    if False:
        while True:
            i = 10
    tools = [Tool(name='ToolA', pipeline_or_node=mock.Mock(), description='Tool A Description'), Tool(name='ToolB', pipeline_or_node=mock.Mock(), description='Tool B Description')]
    return ToolsManager(tools=tools)

@pytest.mark.unit
def test_using_callable_as_tool():
    if False:
        while True:
            i = 10
    tool_input = 'Haystack'
    tool = Tool(name='ToolA', pipeline_or_node=lambda x: x + x, description='Tool A Description')
    assert tool.run(tool_input) == tool_input + tool_input

@pytest.mark.unit
def test_get_tool_names(tools_manager):
    if False:
        while True:
            i = 10
    assert tools_manager.get_tool_names() == 'ToolA, ToolB'

@pytest.mark.unit
def test_get_tools(tools_manager):
    if False:
        return 10
    tools = tools_manager.get_tools()
    assert len(tools) == 2
    assert tools[0].name == 'ToolA'
    assert tools[1].name == 'ToolB'

@pytest.mark.unit
def test_get_tool_names_with_descriptions(tools_manager):
    if False:
        return 10
    expected_output = 'ToolA: Tool A Description\nToolB: Tool B Description'
    assert tools_manager.get_tool_names_with_descriptions() == expected_output

@pytest.mark.unit
def test_extract_tool_name_and_tool_input(tools_manager):
    if False:
        print('Hello World!')
    examples = ['need to find out what city he was born.\nTool: Search\nTool Input: Where was Jeremy McKinnon born', 'need to find out what city he was born.\n\nTool: Search\n\nTool Input: Where was Jeremy McKinnon born', 'need to find out what city he was born. Tool: Search Tool Input: "Where was Jeremy McKinnon born"']
    for example in examples:
        (tool_name, tool_input) = tools_manager.extract_tool_name_and_tool_input(example)
        assert tool_name == 'Search' and tool_input == 'Where was Jeremy McKinnon born'
    negative_examples = ['need to find out what city he was born.', 'Tool: Search', 'Tool Input: Where was Jeremy McKinnon born', 'need to find out what city he was born. Tool: Search', 'Tool Input: Where was Jeremy McKinnon born']
    for example in negative_examples:
        (tool_name, tool_input) = tools_manager.extract_tool_name_and_tool_input(example)
        assert tool_name is None and tool_input is None

@pytest.mark.unit
def test_invalid_tool_creation():
    if False:
        return 10
    with pytest.raises(ValueError, match='Invalid'):
        Tool(name='Tool-A', pipeline_or_node=mock.Mock(), description='Tool A Description')

@pytest.mark.unit
def test_tool_invocation():
    if False:
        for i in range(10):
            print('nop')
    p = Pipeline()
    tool = Tool(name='ToolA', pipeline_or_node=p, description='Tool A Description')
    with unittest.mock.patch('haystack.pipelines.Pipeline.run', return_value={'results': 'mock'}):
        assert tool.run('input') == 'mock'
    with unittest.mock.patch('haystack.pipelines.Pipeline.run', return_value={'no_results': 'mock'}), pytest.raises(ValueError, match='Tool ToolA returned result'):
        assert tool.run('input')
    tool = Tool(name='ToolA', pipeline_or_node=p, description='Tool A Description', output_variable='no_results')
    with unittest.mock.patch('haystack.pipelines.Pipeline.run', return_value={'no_results': 'mock_no_results'}):
        assert tool.run('input') == 'mock_no_results'
    tool = Tool(name='ToolA', pipeline_or_node=p, description='Tool A Description')
    with unittest.mock.patch('haystack.pipelines.Pipeline.run', return_value=[Answer('mocked_answer')]):
        assert tool.run('input') == 'mocked_answer'
    with unittest.mock.patch('haystack.pipelines.Pipeline.run', return_value=[Document('mocked_document')]):
        assert tool.run('input') == 'mocked_document'

@pytest.mark.unit
def test_extract_tool_name_and_tool_multi_line_input(tools_manager):
    if False:
        i = 10
        return i + 15
    text = "We need to find out the following information:\n1. What city was Jeremy McKinnon born in?\n2. What's the capital of Germany?\nTool: Search\nTool Input: Where was Jeremy\n McKinnon born\n and where did he grow up?"
    (tool_name, tool_input) = tools_manager.extract_tool_name_and_tool_input(text)
    assert tool_name == 'Search' and tool_input == 'Where was Jeremy\n McKinnon born\n and where did he grow up?'
    text2 = "We need to find out the following information:\n1. What city was Jeremy McKinnon born in?\n2. What's the capital of Germany?\nTool: Search\nTool Input:"
    (tool_name, tool_input) = tools_manager.extract_tool_name_and_tool_input(text2)
    assert tool_name == 'Search' and tool_input == ''
    text3 = '   Tool:   Search   \n   Tool Input:   What is the tallest building in the world?   '
    (tool_name, tool_input) = tools_manager.extract_tool_name_and_tool_input(text3)
    assert tool_name.strip() == 'Search' and tool_input.strip() == 'What is the tallest building in the world?'
    text4 = 'We need to find out the following information:\n1. Who is the current president of the United States?\nTool: Search\n'
    (tool_name, tool_input) = tools_manager.extract_tool_name_and_tool_input(text4)
    assert tool_name is None and tool_input is None
    text5 = 'We need to find out the following information:\n 1. What is the population of India?'
    (tool_name, tool_input) = tools_manager.extract_tool_name_and_tool_input(text5)
    assert tool_name is None and tool_input is None
    text6 = '   Tool:   Search   \n   Tool Input:   \nWhat is the tallest \nbuilding in the world?   '
    (tool_name, tool_input) = tools_manager.extract_tool_name_and_tool_input(text6)
    assert tool_name.strip() == 'Search' and tool_input.strip() == 'What is the tallest \nbuilding in the world?'

@pytest.mark.unit
def test_extract_tool_name_and_empty_tool_input(tools_manager):
    if False:
        i = 10
        return i + 15
    examples = ['need to find out what city he was born.\nTool: Search\nTool Input:', 'need to find out what city he was born.\nTool: Search\nTool Input:  ']
    for example in examples:
        (tool_name, tool_input) = tools_manager.extract_tool_name_and_tool_input(example)
        assert tool_name == 'Search' and tool_input == ''

@pytest.mark.unit
def test_node_as_tool():
    if False:
        while True:
            i = 10

    class ToolComponent(BaseComponent):
        outgoing_edges = 1

        def run_batch(self, queries: Optional[Union[str, List[str]]]=None, file_paths: Optional[List[str]]=None, labels: Optional[Union[MultiLabel, List[MultiLabel]]]=None, documents: Optional[Union[List[Document], List[List[Document]]]]=None, meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]=None, params: Optional[dict]=None, debug: Optional[bool]=None):
            if False:
                print('Hello World!')
            pass

        def run(self, **kwargs):
            if False:
                print('Hello World!')
            return 'mocked_output'
    tool = Tool(name='ToolA', pipeline_or_node=ToolComponent(), description='Tool A Description')
    assert tool.run('input') == 'mocked_output'

@pytest.mark.unit
def test_tools_manager_exception():
    if False:
        while True:
            i = 10

    class ToolComponent(BaseComponent):
        outgoing_edges = 1

        def run_batch(self, queries: Optional[Union[str, List[str]]]=None, file_paths: Optional[List[str]]=None, labels: Optional[Union[MultiLabel, List[MultiLabel]]]=None, documents: Optional[Union[List[Document], List[List[Document]]]]=None, meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]=None, params: Optional[dict]=None, debug: Optional[bool]=None):
            if False:
                while True:
                    i = 10
            pass

        def run(self, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            raise Exception('mocked_exception')
    fake_llm_response = 'need to find out what city he was born.\nTool: Search\nTool Input: Where was Jeremy born'
    tool = Tool(name='Search', pipeline_or_node=ToolComponent(), description='Search')
    tools_manager = ToolsManager(tools=[tool])
    with pytest.raises(Exception):
        tools_manager.run_tool(llm_response=fake_llm_response)