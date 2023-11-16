import typing
from builtins import _test_sink, _test_source

def test_del_keyword():
    if False:
        while True:
            i = 10
    val = _test_source()
    my_dict = {'key': val}
    del my_dict['key']
    _test_sink(my_dict['key'])

def return_dict_with_bad_key():
    if False:
        print('Hello World!')
    val = _test_source()
    my_dict = {'key': val}
    del my_dict['key']
    return my_dict

def take_dict_with_bad_key(my_dict: typing.Dict[str, str]):
    if False:
        while True:
            i = 10
    del my_dict['key']
    return my_dict['key']

def pop_dict_with_bad_key(my_dict: typing.Dict[str, str]):
    if False:
        return 10
    my_dict.pop('key')
    return my_dict['key']

def pop_key(my_dict: typing.Dict[str, str]):
    if False:
        i = 10
        return i + 15
    return my_dict.pop('key')

def dict_into_sink(my_dict: typing.Dict[str, str]):
    if False:
        return 10
    del my_dict['key']
    _test_sink(my_dict['key'])

def test_pop_method():
    if False:
        i = 10
        return i + 15
    val = _test_source()
    my_dict = {'key': val}
    my_dict.pop('key')
    _test_sink(my_dict['key'])