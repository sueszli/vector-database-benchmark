import pytest
import sacred.optional as opt
from sacred.config import ConfigDict
from sacred.config.custom_containers import DogmaticDict, DogmaticList

@pytest.fixture
def conf_dict():
    if False:
        return 10
    cfg = ConfigDict({'a': 1, 'b': 2.0, 'c': True, 'd': 'string', 'e': [1, 2, 3], 'f': {'a': 'b', 'c': 'd'}})
    return cfg

def test_config_dict_returns_dict(conf_dict):
    if False:
        print('Hello World!')
    assert isinstance(conf_dict(), dict)

def test_config_dict_result_contains_keys(conf_dict):
    if False:
        for i in range(10):
            print('nop')
    cfg = conf_dict()
    assert set(cfg.keys()) == {'a', 'b', 'c', 'd', 'e', 'f'}
    assert cfg['a'] == 1
    assert cfg['b'] == 2.0
    assert cfg['c']
    assert cfg['d'] == 'string'
    assert cfg['e'] == [1, 2, 3]
    assert cfg['f'] == {'a': 'b', 'c': 'd'}

def test_fixing_values(conf_dict):
    if False:
        print('Hello World!')
    assert conf_dict({'a': 100})['a'] == 100

@pytest.mark.parametrize('key', ['$f', 'contains.dot', 'py/tuple', 'json://1'])
def test_config_dict_raises_on_invalid_keys(key):
    if False:
        i = 10
        return i + 15
    with pytest.raises(KeyError):
        ConfigDict({key: True})

@pytest.mark.parametrize('value', [lambda x: x, pytest, test_fixing_values])
def test_config_dict_accepts_special_types(value):
    if False:
        for i in range(10):
            print('nop')
    assert ConfigDict({'special': value})()['special'] == value

def test_fixing_nested_dicts(conf_dict):
    if False:
        return 10
    cfg = conf_dict({'f': {'c': 't'}})
    assert cfg['f']['a'] == 'b'
    assert cfg['f']['c'] == 't'

def test_adding_values(conf_dict):
    if False:
        print('Hello World!')
    cfg = conf_dict({'g': 23, 'h': {'i': 10}})
    assert cfg['g'] == 23
    assert cfg['h'] == {'i': 10}
    assert cfg.added == {'g', 'h', 'h.i'}

def test_typechange(conf_dict):
    if False:
        while True:
            i = 10
    cfg = conf_dict({'a': 'bar', 'b': 'foo', 'c': 1})
    assert cfg.typechanged == {'a': (int, type('bar')), 'b': (float, type('foo')), 'c': (bool, int)}

def test_nested_typechange(conf_dict):
    if False:
        return 10
    cfg = conf_dict({'f': {'a': 10}})
    assert cfg.typechanged == {'f.a': (type('a'), int)}

def is_dogmatic(a):
    if False:
        return 10
    if isinstance(a, (DogmaticDict, DogmaticList)):
        return True
    elif isinstance(a, dict):
        return any((is_dogmatic(v) for v in a.values()))
    elif isinstance(a, (list, tuple)):
        return any((is_dogmatic(v) for v in a))

def test_result_of_conf_dict_is_not_dogmatic(conf_dict):
    if False:
        for i in range(10):
            print('nop')
    cfg = conf_dict({'e': [1, 1, 1]})
    assert not is_dogmatic(cfg)

@pytest.mark.skipif(not opt.has_numpy, reason='requires numpy')
def test_conf_scope_handles_numpy_bools():
    if False:
        while True:
            i = 10
    cfg = ConfigDict({'a': opt.np.bool_(1)})
    assert 'a' in cfg()
    assert cfg()['a']

def test_conf_scope_contains_presets():
    if False:
        print('Hello World!')
    conf_dict = ConfigDict({'answer': 42})
    cfg = conf_dict(preset={'a': 21, 'unrelated': True})
    assert set(cfg.keys()) == {'a', 'answer', 'unrelated'}
    assert cfg['a'] == 21
    assert cfg['answer'] == 42
    assert cfg['unrelated'] is True

def test_conf_scope_does_not_contain_fallback():
    if False:
        return 10
    config_dict = ConfigDict({'answer': 42})
    cfg = config_dict(fallback={'a': 21, 'b': 10})
    assert set(cfg.keys()) == {'answer'}

def test_fixed_subentry_of_preset():
    if False:
        print('Hello World!')
    config_dict = ConfigDict({})
    cfg = config_dict(preset={'d': {'a': 1, 'b': 2}}, fixed={'d': {'a': 10}})
    assert set(cfg.keys()) == {'d'}
    assert set(cfg['d'].keys()) == {'a', 'b'}
    assert cfg['d']['a'] == 10
    assert cfg['d']['b'] == 2

def test_add_config_dict_sequential():
    if False:
        print('Hello World!')
    adict = ConfigDict(dict(dictnest2={'key_1': 'value_1', 'key_2': 'value_2'}))
    bdict = ConfigDict(dict(dictnest2={'key_2': 'update_value_2', 'key_3': 'value3', 'key_4': 'value4'}))
    final_config = bdict(preset=adict())
    assert final_config == {'dictnest2': {'key_1': 'value_1', 'key_2': 'update_value_2', 'key_3': 'value3', 'key_4': 'value4'}}