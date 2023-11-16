import pytest
from unittest.mock import Mock, patch
from superagi.agent.tool_builder import ToolBuilder
from superagi.models.tool import Tool

@pytest.fixture
def session():
    if False:
        while True:
            i = 10
    return Mock()

@pytest.fixture
def agent_id():
    if False:
        for i in range(10):
            print('nop')
    return 1

@pytest.fixture
def tool_builder(session, agent_id):
    if False:
        i = 10
        return i + 15
    return ToolBuilder(session, agent_id)

@pytest.fixture
def tool():
    if False:
        return 10
    tool = Mock(spec=Tool)
    tool.file_name = 'test.py'
    tool.folder_name = 'test_folder'
    tool.class_name = 'TestClass'
    return tool

@pytest.fixture
def agent_config():
    if False:
        while True:
            i = 10
    return {'model': 'gpt4'}

@pytest.fixture
def agent_execution_config():
    if False:
        while True:
            i = 10
    return {'goal': 'Test Goal', 'instruction': 'Test Instruction'}

@patch('superagi.agent.tool_builder.importlib.import_module')
@patch('superagi.agent.tool_builder.getattr')
def test_build_tool(mock_getattr, mock_import_module, tool_builder, tool):
    if False:
        print('Hello World!')
    mock_module = Mock()
    mock_class = Mock()
    mock_import_module.return_value = mock_module
    mock_getattr.return_value = mock_class
    result_tool = tool_builder.build_tool(tool)
    mock_import_module.assert_called_with('.test_folder.test')
    mock_getattr.assert_called_with(mock_module, tool.class_name)
    assert result_tool.toolkit_config.session == tool_builder.session
    assert result_tool.toolkit_config.toolkit_id == tool.toolkit_id