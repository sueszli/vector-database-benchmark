import pytest
from haystack import Answer
from haystack.agents import AgentStep
from haystack.errors import AgentError

@pytest.fixture
def agent_step():
    if False:
        while True:
            i = 10
    return AgentStep(current_step=1, max_steps=10, final_answer_pattern=None, prompt_node_response='Hello', transcript='Hello')

@pytest.mark.unit
def test_create_next_step(agent_step):
    if False:
        for i in range(10):
            print('nop')
    next_step = agent_step.create_next_step(['Hello again'])
    assert next_step.current_step == 2
    assert next_step.prompt_node_response == 'Hello again'
    assert next_step.transcript == 'Hello'
    with pytest.raises(AgentError):
        agent_step.create_next_step({})
    with pytest.raises(AgentError):
        agent_step.create_next_step([])

@pytest.mark.unit
def test_final_answer(agent_step):
    if False:
        for i in range(10):
            print('nop')
    result = agent_step.final_answer('query')
    assert result['query'] == 'query'
    assert isinstance(result['answers'][0], Answer)
    assert result['answers'][0].answer == 'Hello'
    assert result['answers'][0].type == 'generative'
    assert result['transcript'] == 'Hello'
    agent_step.current_step = 11
    result = agent_step.final_answer('query')
    assert result['answers'][0].answer == ''

@pytest.mark.unit
def test_is_last():
    if False:
        return 10
    agent_step = AgentStep(current_step=1, max_steps=10, prompt_node_response='Hello', transcript='Hello')
    assert agent_step.is_last()
    agent_step.current_step = 1
    agent_step.prompt_node_response = 'final answer not satisfying pattern'
    agent_step.final_answer_pattern = 'Final Answer\\s*:\\s*(.*)'
    assert not agent_step.is_last()
    agent_step.current_step = 9
    assert not agent_step.is_last()
    agent_step.current_step = 10
    assert not agent_step.is_last()
    agent_step.current_step = 11
    assert agent_step.is_last()

@pytest.mark.unit
def test_completed(agent_step):
    if False:
        while True:
            i = 10
    agent_step.completed(None)
    assert agent_step.transcript == 'HelloHello'
    agent_step.completed('observation')
    assert agent_step.transcript == 'HelloHelloHello\nObservation: observation\nThought:'

@pytest.mark.unit
def test_repr(agent_step):
    if False:
        for i in range(10):
            print('nop')
    assert repr(agent_step) == 'AgentStep(current_step=1, max_steps=10, prompt_node_response=Hello, final_answer_pattern=^([\\s\\S]+)$, transcript=Hello)'

@pytest.mark.unit
def test_parse_final_answer(agent_step):
    if False:
        return 10
    assert agent_step.parse_final_answer() == 'Hello'
    agent_step.final_answer_pattern = 'goodbye'
    assert agent_step.parse_final_answer() is None

@pytest.mark.unit
def test_format_react_answer():
    if False:
        while True:
            i = 10
    step = AgentStep(final_answer_pattern='Final Answer\\s*:\\s*(.*)', prompt_node_response='have the final answer to the question.\nFinal Answer: Florida')
    formatted_answer = step.final_answer(query='query')
    assert formatted_answer['query'] == 'query'
    assert formatted_answer['answers'] == [Answer(answer='Florida', type='generative')]