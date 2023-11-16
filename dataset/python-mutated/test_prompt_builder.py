import pytest
from haystack.preview.components.builders.prompt_builder import PromptBuilder
from haystack.preview.dataclasses import ChatMessage

@pytest.mark.unit
def test_init():
    if False:
        while True:
            i = 10
    builder = PromptBuilder(template='This is a {{ variable }}')
    assert builder._template_string == 'This is a {{ variable }}'

@pytest.mark.unit
def test_to_dict():
    if False:
        while True:
            i = 10
    builder = PromptBuilder(template='This is a {{ variable }}')
    res = builder.to_dict()
    assert res == {'type': 'PromptBuilder', 'init_parameters': {'template': 'This is a {{ variable }}', 'template_variables': None}}

@pytest.mark.unit
def test_run():
    if False:
        i = 10
        return i + 15
    builder = PromptBuilder(template='This is a {{ variable }}')
    res = builder.run(variable='test')
    assert res == {'prompt': 'This is a test'}

@pytest.mark.unit
def test_run_without_input():
    if False:
        i = 10
        return i + 15
    builder = PromptBuilder(template='This is a template without input')
    res = builder.run()
    assert res == {'prompt': 'This is a template without input'}

@pytest.mark.unit
def test_run_with_missing_input():
    if False:
        print('Hello World!')
    builder = PromptBuilder(template='This is a {{ variable }}')
    res = builder.run()
    assert res == {'prompt': 'This is a '}

@pytest.mark.unit
def test_init_with_template_and_template_variables():
    if False:
        return 10
    with pytest.raises(ValueError, match='template and template_variables cannot be provided at the same time.'):
        PromptBuilder(template='This is a {{ variable }}', template_variables=['variable'])

@pytest.mark.unit
def test_init_with_no_template_and_no_template_variables():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match='Either template or template_variables must be provided.'):
        PromptBuilder()

@pytest.mark.unit
def test_dynamic_template_with_input_variables_no_messages():
    if False:
        print('Hello World!')
    template_variables = ['location', 'time']
    builder = PromptBuilder(template_variables=template_variables)
    with pytest.raises(ValueError, match='PromptBuilder was initialized with template_variables'):
        builder.run(location='New York', time='tomorrow')

@pytest.mark.unit
def test_dynamic_template_with_input_variables_and_messages():
    if False:
        while True:
            i = 10
    template_variables = ['location', 'time']
    builder = PromptBuilder(template_variables=template_variables)
    system_message = 'Always start response to user with Herr Blagojevic. Respond in German even if some input data is in other languages'
    messages = [ChatMessage.from_system(system_message), ChatMessage.from_user("What's the weather like in {{ location }}?")]
    result = builder.run(messages=messages, location='New York', time='tomorrow')
    assert result['prompt'] == [ChatMessage.from_system(system_message), ChatMessage.from_user("What's the weather like in New York?")]

@pytest.mark.unit
def test_static_template_without_input_variables():
    if False:
        i = 10
        return i + 15
    template = 'Translate the following context to Spanish.'
    builder = PromptBuilder(template=template)
    result = builder.run()
    assert result['prompt'] == 'Translate the following context to Spanish.'

@pytest.mark.unit
def test_dynamic_template_without_input_variables():
    if False:
        return 10
    template_variables = ['location', 'time']
    builder = PromptBuilder(template_variables=template_variables)
    messages = [ChatMessage.from_user("What's LLM?")]
    result = builder.run(messages=messages)
    assert result['prompt'] == [ChatMessage.from_user("What's LLM?")]

@pytest.mark.unit
def test_dynamic_template_with_input_variables_and_multiple_user_messages():
    if False:
        i = 10
        return i + 15
    template_variables = ['location', 'time']
    builder = PromptBuilder(template_variables=template_variables)
    system_message = 'Always start response to user with Herr Blagojevic. Respond in German even if some input data is in other languages'
    messages = [ChatMessage.from_system(system_message), ChatMessage.from_user('Here is improper use of {{ location }} as it is not the last message'), ChatMessage.from_user("What's the weather like in {{ location }}?")]
    result = builder.run(messages=messages, location='New York', time='tomorrow')
    assert result['prompt'] == [ChatMessage.from_system(system_message), ChatMessage.from_user('Here is improper use of {{ location }} as it is not the last message'), ChatMessage.from_user("What's the weather like in New York?")]

def test_dynamic_template_with_invalid_input_variables_and_messages():
    if False:
        for i in range(10):
            print('nop')
    template_variables = ['location', 'time']
    builder = PromptBuilder(template_variables=template_variables)
    system_message = 'Always start response to user with Herr Blagojevic. Respond in German even if some input data is in other languages'
    messages = [ChatMessage.from_system(system_message), ChatMessage.from_user('What is {{ topic }}?')]
    result = builder.run(messages=messages, location='New York', time='tomorrow')
    assert result['prompt'] == [ChatMessage.from_system(system_message), ChatMessage.from_user('What is ?')]