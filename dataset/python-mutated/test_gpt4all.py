import pytest
from langchain.llms.gpt4all import GPT4All as LangchainGPT4All
from embedchain.config import BaseLlmConfig
from embedchain.llm.gpt4all import GPT4ALLLlm

@pytest.fixture
def config():
    if False:
        return 10
    config = BaseLlmConfig(temperature=0.7, max_tokens=50, top_p=0.8, stream=False, system_prompt='System prompt', model='orca-mini-3b-gguf2-q4_0.gguf')
    yield config

@pytest.fixture
def gpt4all_with_config(config):
    if False:
        while True:
            i = 10
    return GPT4ALLLlm(config=config)

@pytest.fixture
def gpt4all_without_config():
    if False:
        i = 10
        return i + 15
    return GPT4ALLLlm()

def test_gpt4all_init_with_config(config, gpt4all_with_config):
    if False:
        print('Hello World!')
    assert gpt4all_with_config.config.temperature == config.temperature
    assert gpt4all_with_config.config.max_tokens == config.max_tokens
    assert gpt4all_with_config.config.top_p == config.top_p
    assert gpt4all_with_config.config.stream == config.stream
    assert gpt4all_with_config.config.system_prompt == config.system_prompt
    assert gpt4all_with_config.config.model == config.model
    assert isinstance(gpt4all_with_config.instance, LangchainGPT4All)

def test_gpt4all_init_without_config(gpt4all_without_config):
    if False:
        print('Hello World!')
    assert gpt4all_without_config.config.model == 'orca-mini-3b-gguf2-q4_0.gguf'
    assert isinstance(gpt4all_without_config.instance, LangchainGPT4All)

def test_get_llm_model_answer(mocker, gpt4all_with_config):
    if False:
        while True:
            i = 10
    test_query = 'Test query'
    test_answer = 'Test answer'
    mocked_get_answer = mocker.patch('embedchain.llm.gpt4all.GPT4ALLLlm._get_answer', return_value=test_answer)
    answer = gpt4all_with_config.get_llm_model_answer(test_query)
    assert answer == test_answer
    mocked_get_answer.assert_called_once_with(prompt=test_query, config=gpt4all_with_config.config)

def test_gpt4all_model_switching(gpt4all_with_config):
    if False:
        print('Hello World!')
    with pytest.raises(RuntimeError, match='GPT4ALLLlm does not support switching models at runtime.'):
        gpt4all_with_config._get_answer('Test prompt', BaseLlmConfig(model='new_model'))