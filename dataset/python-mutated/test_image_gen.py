import functools
import hashlib
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image
from autogpt.agents.agent import Agent
from autogpt.commands.image_gen import generate_image, generate_image_with_sd_webui

@pytest.fixture(params=[256, 512, 1024])
def image_size(request):
    if False:
        for i in range(10):
            print('nop')
    'Parametrize image size.'
    return request.param

@pytest.mark.requires_openai_api_key
@pytest.mark.vcr
def test_dalle(agent: Agent, workspace, image_size, patched_api_requestor):
    if False:
        for i in range(10):
            print('nop')
    'Test DALL-E image generation.'
    generate_and_validate(agent, workspace, image_provider='dalle', image_size=image_size)

@pytest.mark.xfail(reason="The image is too big to be put in a cassette for a CI pipeline. We're looking into a solution.")
@pytest.mark.requires_huggingface_api_key
@pytest.mark.parametrize('image_model', ['CompVis/stable-diffusion-v1-4', 'stabilityai/stable-diffusion-2-1'])
def test_huggingface(agent: Agent, workspace, image_size, image_model):
    if False:
        print('Hello World!')
    'Test HuggingFace image generation.'
    generate_and_validate(agent, workspace, image_provider='huggingface', image_size=image_size, hugging_face_image_model=image_model)

@pytest.mark.xfail(reason='SD WebUI call does not work.')
def test_sd_webui(agent: Agent, workspace, image_size):
    if False:
        i = 10
        return i + 15
    'Test SD WebUI image generation.'
    generate_and_validate(agent, workspace, image_provider='sd_webui', image_size=image_size)

@pytest.mark.xfail(reason='SD WebUI call does not work.')
def test_sd_webui_negative_prompt(agent: Agent, workspace, image_size):
    if False:
        print('Hello World!')
    gen_image = functools.partial(generate_image_with_sd_webui, prompt='astronaut riding a horse', agent=agent, size=image_size, extra={'seed': 123})
    image_path = lst(gen_image(negative_prompt='horse', filename='negative.jpg'))
    with Image.open(image_path) as img:
        neg_image_hash = hashlib.md5(img.tobytes()).hexdigest()
    image_path = lst(gen_image(filename='positive.jpg'))
    with Image.open(image_path) as img:
        image_hash = hashlib.md5(img.tobytes()).hexdigest()
    assert image_hash != neg_image_hash

def lst(txt):
    if False:
        print('Hello World!')
    'Extract the file path from the output of `generate_image()`'
    return Path(txt.split(':', maxsplit=1)[1].strip())

def generate_and_validate(agent: Agent, workspace, image_size, image_provider, hugging_face_image_model=None, **kwargs):
    if False:
        while True:
            i = 10
    'Generate an image and validate the output.'
    agent.legacy_config.image_provider = image_provider
    agent.legacy_config.huggingface_image_model = hugging_face_image_model
    prompt = 'astronaut riding a horse'
    image_path = lst(generate_image(prompt, agent, image_size, **kwargs))
    assert image_path.exists()
    with Image.open(image_path) as img:
        assert img.size == (image_size, image_size)

@pytest.mark.parametrize('return_text', ['{"error":"Model [model] is currently loading","estimated_time": [delay]}', '{"error":"Model [model] is currently loading"}', '{"error:}', ''])
@pytest.mark.parametrize('image_model', ['CompVis/stable-diffusion-v1-4', 'stabilityai/stable-diffusion-2-1'])
@pytest.mark.parametrize('delay', [10, 0])
def test_huggingface_fail_request_with_delay(agent: Agent, workspace, image_size, image_model, return_text, delay):
    if False:
        for i in range(10):
            print('nop')
    return_text = return_text.replace('[model]', image_model).replace('[delay]', str(delay))
    with patch('requests.post') as mock_post:
        if return_text == '':
            mock_post.return_value.status_code = 200
            mock_post.return_value.ok = True
            mock_post.return_value.content = b'bad image'
        else:
            mock_post.return_value.status_code = 500
            mock_post.return_value.ok = False
            mock_post.return_value.text = return_text
        agent.legacy_config.image_provider = 'huggingface'
        agent.legacy_config.huggingface_image_model = image_model
        prompt = 'astronaut riding a horse'
        with patch('time.sleep') as mock_sleep:
            result = generate_image(prompt, agent, image_size)
            assert result == 'Error creating image.'
            if 'estimated_time' in return_text:
                mock_sleep.assert_called_with(delay)
            else:
                mock_sleep.assert_not_called()

def test_huggingface_fail_request_with_delay(mocker, agent: Agent):
    if False:
        print('Hello World!')
    agent.legacy_config.huggingface_api_token = '1'
    mock_post = mocker.patch('requests.post')
    mock_post.return_value.status_code = 500
    mock_post.return_value.ok = False
    mock_post.return_value.text = '{"error":"Model CompVis/stable-diffusion-v1-4 is currently loading","estimated_time":0}'
    mock_sleep = mocker.patch('time.sleep')
    agent.legacy_config.image_provider = 'huggingface'
    agent.legacy_config.huggingface_image_model = 'CompVis/stable-diffusion-v1-4'
    result = generate_image('astronaut riding a horse', agent, 512)
    assert result == 'Error creating image.'
    mock_sleep.assert_called_with(0)

def test_huggingface_fail_request_no_delay(mocker, agent: Agent):
    if False:
        return 10
    agent.legacy_config.huggingface_api_token = '1'
    mock_post = mocker.patch('requests.post')
    mock_post.return_value.status_code = 500
    mock_post.return_value.ok = False
    mock_post.return_value.text = '{"error":"Model CompVis/stable-diffusion-v1-4 is currently loading"}'
    mock_sleep = mocker.patch('time.sleep')
    agent.legacy_config.image_provider = 'huggingface'
    agent.legacy_config.huggingface_image_model = 'CompVis/stable-diffusion-v1-4'
    result = generate_image('astronaut riding a horse', agent, 512)
    assert result == 'Error creating image.'
    mock_sleep.assert_not_called()

def test_huggingface_fail_request_bad_json(mocker, agent: Agent):
    if False:
        print('Hello World!')
    agent.legacy_config.huggingface_api_token = '1'
    mock_post = mocker.patch('requests.post')
    mock_post.return_value.status_code = 500
    mock_post.return_value.ok = False
    mock_post.return_value.text = '{"error:}'
    mock_sleep = mocker.patch('time.sleep')
    agent.legacy_config.image_provider = 'huggingface'
    agent.legacy_config.huggingface_image_model = 'CompVis/stable-diffusion-v1-4'
    result = generate_image('astronaut riding a horse', agent, 512)
    assert result == 'Error creating image.'
    mock_sleep.assert_not_called()

def test_huggingface_fail_request_bad_image(mocker, agent: Agent):
    if False:
        return 10
    agent.legacy_config.huggingface_api_token = '1'
    mock_post = mocker.patch('requests.post')
    mock_post.return_value.status_code = 200
    agent.legacy_config.image_provider = 'huggingface'
    agent.legacy_config.huggingface_image_model = 'CompVis/stable-diffusion-v1-4'
    result = generate_image('astronaut riding a horse', agent, 512)
    assert result == 'Error creating image.'

def test_huggingface_fail_missing_api_token(mocker, agent: Agent):
    if False:
        for i in range(10):
            print('nop')
    agent.legacy_config.image_provider = 'huggingface'
    agent.legacy_config.huggingface_image_model = 'CompVis/stable-diffusion-v1-4'
    mock_post = mocker.patch('requests.post', side_effect=ValueError)
    with pytest.raises(ValueError):
        generate_image('astronaut riding a horse', agent, 512)