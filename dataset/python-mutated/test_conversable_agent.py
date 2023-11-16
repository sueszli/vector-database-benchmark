import pytest
from flaml.autogen.agentchat import ConversableAgent

def test_trigger():
    if False:
        for i in range(10):
            print('nop')
    agent = ConversableAgent('a0', max_consecutive_auto_reply=0, llm_config=False, human_input_mode='NEVER')
    agent1 = ConversableAgent('a1', max_consecutive_auto_reply=0, human_input_mode='NEVER')
    agent.register_reply(agent1, lambda recipient, messages, sender, config: (True, 'hello'))
    agent1.initiate_chat(agent, message='hi')
    assert agent1.last_message(agent)['content'] == 'hello'
    agent.register_reply('a1', lambda recipient, messages, sender, config: (True, 'hello a1'))
    agent1.initiate_chat(agent, message='hi')
    assert agent1.last_message(agent)['content'] == 'hello a1'
    agent.register_reply(ConversableAgent, lambda recipient, messages, sender, config: (True, 'hello conversable agent'))
    agent1.initiate_chat(agent, message='hi')
    assert agent1.last_message(agent)['content'] == 'hello conversable agent'
    agent.register_reply(lambda sender: sender.name.startswith('a'), lambda recipient, messages, sender, config: (True, 'hello a'))
    agent1.initiate_chat(agent, message='hi')
    assert agent1.last_message(agent)['content'] == 'hello a'
    agent.register_reply(lambda sender: sender.name.startswith('b'), lambda recipient, messages, sender, config: (True, 'hello b'))
    agent1.initiate_chat(agent, message='hi')
    assert agent1.last_message(agent)['content'] == 'hello a'
    agent.register_reply(['agent2', agent1], lambda recipient, messages, sender, config: (True, 'hello agent2 or agent1'))
    agent1.initiate_chat(agent, message='hi')
    assert agent1.last_message(agent)['content'] == 'hello agent2 or agent1'
    agent.register_reply(['agent2', 'agent3'], lambda recipient, messages, sender, config: (True, 'hello agent2 or agent3'))
    agent1.initiate_chat(agent, message='hi')
    assert agent1.last_message(agent)['content'] == 'hello agent2 or agent1'
    pytest.raises(ValueError, agent.register_reply, 1, lambda recipient, messages, sender, config: (True, 'hi'))
    pytest.raises(ValueError, agent._match_trigger, 1, agent1)

def test_context():
    if False:
        while True:
            i = 10
    agent = ConversableAgent('a0', max_consecutive_auto_reply=0, llm_config=False, human_input_mode='NEVER')
    agent1 = ConversableAgent('a1', max_consecutive_auto_reply=0, human_input_mode='NEVER')
    agent1.send({'content': 'hello {name}', 'context': {'name': 'there'}}, agent)
    agent1.send({'content': lambda context: f"hello {context['name']}", 'context': {'name': 'there'}}, agent)
    agent.llm_config = {'allow_format_str_template': True}
    agent1.send({'content': 'hello {name}', 'context': {'name': 'there'}}, agent)

def test_max_consecutive_auto_reply():
    if False:
        return 10
    agent = ConversableAgent('a0', max_consecutive_auto_reply=2, llm_config=False, human_input_mode='NEVER')
    agent1 = ConversableAgent('a1', max_consecutive_auto_reply=0, human_input_mode='NEVER')
    assert agent.max_consecutive_auto_reply() == agent.max_consecutive_auto_reply(agent1) == 2
    agent.update_max_consecutive_auto_reply(1)
    assert agent.max_consecutive_auto_reply() == agent.max_consecutive_auto_reply(agent1) == 1
    agent1.initiate_chat(agent, message='hello')
    assert agent._consecutive_auto_reply_counter[agent1] == 1
    agent1.initiate_chat(agent, message='hello again')
    assert agent1.last_message(agent)['role'] == 'user'
    assert len(agent1.chat_messages[agent]) == 2
    assert len(agent.chat_messages[agent1]) == 2
    assert agent._consecutive_auto_reply_counter[agent1] == 1
    agent1.send(message='bye', recipient=agent)
    assert agent1.last_message(agent)['role'] == 'assistant'
    agent1.initiate_chat(agent, clear_history=False, message='hi')
    assert len(agent1.chat_messages[agent]) > 2
    assert len(agent.chat_messages[agent1]) > 2
    assert agent1.reply_at_receive[agent] == agent.reply_at_receive[agent1] is True
    agent1.stop_reply_at_receive(agent)
    assert agent1.reply_at_receive[agent] is False and agent.reply_at_receive[agent1] is True

def test_conversable_agent():
    if False:
        return 10
    dummy_agent_1 = ConversableAgent(name='dummy_agent_1', human_input_mode='ALWAYS')
    dummy_agent_2 = ConversableAgent(name='dummy_agent_2', human_input_mode='TERMINATE')
    dummy_agent_1.receive('hello', dummy_agent_2)
    dummy_agent_1.receive({'content': 'hello {name}', 'context': {'name': 'dummy_agent_2'}}, dummy_agent_2)
    assert 'context' in dummy_agent_1.chat_messages[dummy_agent_2][-1]
    pre_len = len(dummy_agent_1.chat_messages[dummy_agent_2])
    with pytest.raises(ValueError):
        dummy_agent_1.receive({'message': 'hello'}, dummy_agent_2)
    assert pre_len == len(dummy_agent_1.chat_messages[dummy_agent_2]), 'When the message is not an valid openai message, it should not be appended to the oai conversation.'
    dummy_agent_1.send('TERMINATE', dummy_agent_2)
    dummy_agent_1.send({'content': 'TERMINATE'}, dummy_agent_2)
    pre_len = len(dummy_agent_1.chat_messages[dummy_agent_2])
    with pytest.raises(ValueError):
        dummy_agent_1.send({'message': 'hello'}, dummy_agent_2)
    assert pre_len == len(dummy_agent_1.chat_messages[dummy_agent_2]), 'When the message is not a valid openai message, it should not be appended to the oai conversation.'
    dummy_agent_1.update_system_message('new system message')
    assert dummy_agent_1.system_message == 'new system message'

def test_generate_reply():
    if False:
        i = 10
        return i + 15

    def add_num(num_to_be_added):
        if False:
            print('Hello World!')
        given_num = 10
        return num_to_be_added + given_num
    dummy_agent_2 = ConversableAgent(name='user_proxy', human_input_mode='TERMINATE', function_map={'add_num': add_num})
    messsages = [{'function_call': {'name': 'add_num', 'arguments': '{ "num_to_be_added": 5 }'}, 'role': 'assistant'}]
    assert dummy_agent_2.generate_reply(messages=messsages, sender=None)['content'] == '15', 'generate_reply not working when sender is None'
    dummy_agent_1 = ConversableAgent(name='dummy_agent_1', human_input_mode='ALWAYS')
    dummy_agent_2._oai_messages[dummy_agent_1] = messsages
    assert dummy_agent_2.generate_reply(messages=None, sender=dummy_agent_1)['content'] == '15', 'generate_reply not working when messages is None'
if __name__ == '__main__':
    test_trigger()