import pytest
from haystack.preview.components.generators.hf_utils import check_generation_params

@pytest.mark.unit
def test_empty_dictionary():
    if False:
        print('Hello World!')
    check_generation_params({})

@pytest.mark.unit
def test_valid_generation_parameters():
    if False:
        while True:
            i = 10
    kwargs = {'max_new_tokens': 100, 'temperature': 0.8}
    additional_accepted_params = None
    check_generation_params(kwargs, additional_accepted_params)

@pytest.mark.unit
def test_invalid_generation_parameters():
    if False:
        for i in range(10):
            print('nop')
    kwargs = {'invalid_param': 'value'}
    additional_accepted_params = None
    with pytest.raises(ValueError):
        check_generation_params(kwargs, additional_accepted_params)

@pytest.mark.unit
def test_additional_accepted_params_empty_list():
    if False:
        for i in range(10):
            print('nop')
    kwargs = {'temperature': 0.8}
    additional_accepted_params = []
    check_generation_params(kwargs, additional_accepted_params)

@pytest.mark.unit
def test_additional_accepted_params_known_parameter():
    if False:
        print('Hello World!')
    kwargs = {'temperature': 0.8}
    additional_accepted_params = ['max_new_tokens']
    check_generation_params(kwargs, additional_accepted_params)

@pytest.mark.unit
def test_additional_accepted_params_unknown_parameter():
    if False:
        print('Hello World!')
    kwargs = {'strange_param': 'value'}
    additional_accepted_params = ['strange_param']
    check_generation_params(kwargs, additional_accepted_params)