import pytest
from unittest.mock import MagicMock, patch
from haystack.errors import AgentError
from haystack.agents.base import Tool
from haystack.agents.conversational import ConversationalAgent
from haystack.agents.memory import ConversationSummaryMemory, ConversationMemory, NoMemory
from haystack.nodes import PromptNode

@pytest.fixture
@patch('haystack.nodes.prompt.prompt_node.PromptModel')
def prompt_node(mock_model):
    if False:
        print('Hello World!')
    prompt_node = PromptNode()
    return prompt_node

@pytest.mark.unit
def test_init_without_tools(prompt_node):
    if False:
        return 10
    agent = ConversationalAgent(prompt_node)
    assert isinstance(agent.memory, ConversationMemory)
    assert callable(agent.prompt_parameters_resolver)
    assert agent.max_steps == 2
    assert agent.final_answer_pattern == '^([\\s\\S]+)$'
    assert agent.prompt_template.name == 'conversational-agent-without-tools'
    assert not agent.tm.tools

@pytest.mark.unit
def test_init_with_tools(prompt_node):
    if False:
        while True:
            i = 10
    agent = ConversationalAgent(prompt_node, tools=[Tool('ExampleTool', lambda x: x, description='Example tool')])
    assert isinstance(agent.memory, ConversationMemory)
    assert callable(agent.prompt_parameters_resolver)
    assert agent.max_steps == 5
    assert agent.final_answer_pattern == 'Final Answer\\s*:\\s*(.*)'
    assert agent.prompt_template.name == 'conversational-agent'
    assert agent.has_tool('ExampleTool')

@pytest.mark.unit
def test_init_with_summary_memory(prompt_node):
    if False:
        i = 10
        return i + 15
    agent = ConversationalAgent(prompt_node, memory=ConversationSummaryMemory(prompt_node))
    assert isinstance(agent.memory, ConversationSummaryMemory)

@pytest.mark.unit
def test_init_with_no_memory(prompt_node):
    if False:
        for i in range(10):
            print('nop')
    agent = ConversationalAgent(prompt_node, memory=NoMemory())
    assert isinstance(agent.memory, NoMemory)

@pytest.mark.unit
def test_init_with_custom_max_steps(prompt_node):
    if False:
        for i in range(10):
            print('nop')
    agent = ConversationalAgent(prompt_node, max_steps=8)
    assert agent.max_steps == 8

@pytest.mark.unit
def test_init_with_custom_prompt_template(prompt_node):
    if False:
        for i in range(10):
            print('nop')
    agent = ConversationalAgent(prompt_node, prompt_template='translation')
    assert agent.prompt_template.name == 'translation'

@pytest.mark.unit
def test_run(prompt_node):
    if False:
        for i in range(10):
            print('nop')
    agent = ConversationalAgent(prompt_node)
    agent.run = MagicMock(return_value='Hello')
    assert agent.run('query') == 'Hello'
    agent.run.assert_called_once_with('query')

@pytest.mark.unit
def test_add_tool(prompt_node):
    if False:
        print('Hello World!')
    agent = ConversationalAgent(prompt_node, tools=[Tool('ExampleTool', lambda x: x, description='Example tool')])
    assert len(agent.tm.tools) == 1
    agent.add_tool(Tool('AnotherTool', lambda x: x, description='Example tool'))
    assert len(agent.tm.tools) == 2

@pytest.mark.unit
def test_add_tool_not_allowed(prompt_node):
    if False:
        return 10
    agent = ConversationalAgent(prompt_node)
    assert not agent.tm.tools
    with pytest.raises(AgentError, match='You cannot add tools after initializing the ConversationalAgent without any tools.'):
        agent.add_tool(Tool('ExampleTool', lambda x: x, description='Example tool'))