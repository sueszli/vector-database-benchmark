import pytest
import salt.serializers.yamlex as yamlex
import salt.state
from salt.config import minion_config
from salt.template import compile_template_str

def render(template, opts=None):
    if False:
        print('Hello World!')
    _config = minion_config(None)
    _config['file_client'] = 'local'
    if opts:
        _config.update(opts)
    _state = salt.state.State(_config)
    return compile_template_str(template, _state.rend, _state.opts['renderer'], _state.opts['renderer_blacklist'], _state.opts['renderer_whitelist'])

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {yamlex: {}}

@pytest.mark.skipif(yamlex.available is False, reason='yamlex is unavailable, do prerequisites have been met?')
def test_basic():
    if False:
        return 10
    basic_template = '#!yamlex\n    foo: bar\n    '
    sls_obj = render(basic_template)
    assert sls_obj == {'foo': 'bar'}, sls_obj

@pytest.mark.skipif(yamlex.available is False, reason='yamlex is unavailable, do prerequisites have been met?')
def test_complex():
    if False:
        for i in range(10):
            print('nop')
    complex_template = '#!yamlex\n    placeholder: {foo: !aggregate {foo: 42}}\n    placeholder: {foo: !aggregate {bar: null}}\n    placeholder: {foo: !aggregate {baz: inga}}\n    '
    sls_obj = render(complex_template)
    assert sls_obj == {'placeholder': {'foo': {'foo': 42, 'bar': None, 'baz': 'inga'}}}, sls_obj