import pytest
from superagi.agent.output_parser import AgentGPTAction, AgentSchemaOutputParser
import pytest

def test_agent_schema_output_parser():
    if False:
        print('Hello World!')
    parser = AgentSchemaOutputParser()
    response = '```{"tool": {"name": "Tool1", "args": {}}}```'
    parsed = parser.parse(response)
    assert isinstance(parsed, AgentGPTAction)
    assert parsed.name == 'Tool1'
    assert parsed.args == {}
    response = "```{'tool': {'name': 'Tool1', 'args': 'arg1'}, 'status': True}```"
    parsed = parser.parse(response)
    assert isinstance(parsed, AgentGPTAction)
    assert parsed.name == 'Tool1'
    assert parsed.args == 'arg1'
    response = 'invalid response'
    with pytest.raises(Exception):
        parsed = parser.parse(response)
    response = ''
    with pytest.raises(Exception):
        parsed = parser.parse(response)