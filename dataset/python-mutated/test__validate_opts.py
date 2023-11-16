"""
Test config option type enforcement
"""
import pytest
import salt.config

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], True), ((1, 2, 3), True), ({'key': 'value'}, False), ('str', False), (True, False), (1, False), (0.123, False), (None, False)])
def test_list_types(option_value, expected):
    if False:
        i = 10
        return i + 15
    '\n    List and tuple type config options return True when the value is a list. All\n    other types return False\n    modules_dirs is a list type config option\n    '
    result = salt.config._validate_opts({'module_dirs': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, False), ('str', True), (True, True), (1, True), (0.123, True), (None, True)])
def test_str_types(option_value, expected):
    if False:
        return 10
    '\n    Str, bool, int, float, and none type config options return True when the\n    value is a str. All other types return False\n    user is a str type config option\n    '
    result = salt.config._validate_opts({'user': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, True), ('str', False), (True, False), (1, False), (0.123, False), (None, False)])
def test_dict_types(option_value, expected):
    if False:
        print('Hello World!')
    '\n    Dict type config options return True when the value is a dict. All other\n    types return False\n    file_roots is a dict type config option\n    '
    result = salt.config._validate_opts({'file_roots': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, False), ('str', False), (True, True), (1, False), (0.123, False), (None, False)])
def test_bool_types(option_value, expected):
    if False:
        i = 10
        return i + 15
    '\n    Bool type config options return True when the value is a bool. All other\n    types return False\n    local is a bool type config option\n    '
    result = salt.config._validate_opts({'local': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, False), ('str', False), (True, False), (1, True), (0.123, False), (None, False)])
def test_int_types(option_value, expected):
    if False:
        i = 10
        return i + 15
    '\n    Int type config options return True when the value is an int. All other\n    types return False\n    publish_port is an int type config option\n    '
    result = salt.config._validate_opts({'publish_port': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, False), ('str', False), (True, False), (1, True), (0.123, True), (None, False)])
def test_float_types(option_value, expected):
    if False:
        for i in range(10):
            print('nop')
    '\n    Float and int type config options return True when the value is a float. All\n    other types return False\n    ssh_timeout is a float type config option\n    '
    result = salt.config._validate_opts({'ssh_timeout': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, False), ('str', True), (True, True), (1, True), (0.123, True), (None, True)])
def test_none_str_types(option_value, expected):
    if False:
        return 10
    '\n    Some config settings have two types, None and str. In that case str, bool,\n    int, float, and None type options should evaluate as True. All others should\n    return False.\n    saltenv is a None, str type config option\n    '
    result = salt.config._validate_opts({'saltenv': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, False), ('str', False), (True, False), (1, True), (0.123, False), (None, True)])
def test_none_int_types(option_value, expected):
    if False:
        for i in range(10):
            print('nop')
    '\n    Some config settings have two types, None and int, which should evaluate as\n    True. All others should return False.\n    retry_dns_count is a None, int type config option\n    '
    result = salt.config._validate_opts({'retry_dns_count': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, False), ('str', False), (True, True), (1, False), (0.123, False), (None, True)])
def test_none_bool_types(option_value, expected):
    if False:
        return 10
    '\n    Some config settings have two types, None and bool which should evaluate as\n    True. All others should return False.\n    ipv6 is a None, bool type config option\n    '
    result = salt.config._validate_opts({'ipv6': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], True), ((1, 2, 3), True), ({'key': 'value'}, False), ('str', True), (True, True), (1, True), (0.123, True), (None, True)])
def test_str_list_types(option_value, expected):
    if False:
        for i in range(10):
            print('nop')
    '\n    Some config settings have two types, str and list. In that case, list,\n    tuple, str, bool, int, float, and None should evaluate as True. All others\n    should return False.\n    master is a str, list type config option\n    '
    result = salt.config._validate_opts({'master': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, False), ('str', True), (True, True), (1, True), (0.123, True), (None, True)])
def test_str_int_types(option_value, expected):
    if False:
        for i in range(10):
            print('nop')
    '\n    Some config settings have two types, str and int. In that case, str, bool,\n    int, float, and None should evaluate as True. All others should return\n    False.\n    master_port is a str, int type config option\n    '
    result = salt.config._validate_opts({'master_port': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, True), ('str', True), (True, True), (1, True), (0.123, True), (None, True)])
def test_str_dict_types(option_value, expected):
    if False:
        for i in range(10):
            print('nop')
    '\n    Some config settings have two types, str and dict. In that case, dict, str,\n    bool, int, float, and None should evaluate as True. All others should return\n    False.\n    id_function is a str, dict type config option\n    '
    result = salt.config._validate_opts({'id_function': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], True), ((1, 2, 3), True), ({'key': 'value'}, False), ('str', True), (True, True), (1, True), (0.123, True), (None, True)])
def test_str_tuple_types(option_value, expected):
    if False:
        i = 10
        return i + 15
    '\n    Some config settings have two types, str and tuple. In that case, list,\n    tuple, str, bool, int, float, and None should evaluate as True. All others\n    should return False.\n    log_fmt_logfile is a str, tuple type config option\n    '
    result = salt.config._validate_opts({'log_fmt_logfile': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, False), ('str', True), (True, True), (1, True), (0.123, True), (None, True)])
def test_str_bool_types(option_value, expected):
    if False:
        while True:
            i = 10
    '\n    Some config settings have two types, str and bool. In that case, str, bool,\n    int, float, and None should evaluate as True. All others should return\n    False.\n    update_url is a str, bool type config option\n    '
    result = salt.config._validate_opts({'update_url': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, True), ('str', False), (True, True), (1, False), (0.123, False), (None, False)])
def test_dict_bool_types(option_value, expected):
    if False:
        i = 10
        return i + 15
    '\n    Some config settings have two types, dict and bool which should evaluate as\n    True. All others should return False.\n    token_expire_user_override is a dict, bool type config option\n    '
    result = salt.config._validate_opts({'token_expire_user_override': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], True), ((1, 2, 3), True), ({'key': 'value'}, True), ('str', False), (True, False), (1, False), (0.123, False), (None, False)])
def test_dict_list_types(option_value, expected):
    if False:
        i = 10
        return i + 15
    '\n    Some config settings have two types, dict and list. In that case, list,\n    tuple, and dict should evaluate as True. All others should return False.\n    nodegroups is a dict, list type config option\n    '
    result = salt.config._validate_opts({'nodegroups': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, True), ('str', False), (True, True), (1, False), (0.123, False), (None, True)])
def test_dict_bool_none_types(option_value, expected):
    if False:
        print('Hello World!')
    '\n    Some config settings have three types, dict, bool, and None which should\n    evaluate as True. All others should return False.\n    ssl is a dict, bool type config option\n    '
    result = salt.config._validate_opts({'ssl': option_value})
    assert result is expected

@pytest.mark.parametrize('option_value,expected', [([1, 2, 3], False), ((1, 2, 3), False), ({'key': 'value'}, False), ('str', False), (True, True), (1, True), (0.123, False), (None, False)])
def test_bool_int_types(option_value, expected):
    if False:
        for i in range(10):
            print('nop')
    '\n    Some config settings have two types, bool and int. In that case, bool and\n    int should evaluate as True. All others should return False.\n    state_queue is a bool/int config option\n    '
    result = salt.config._validate_opts({'state_queue': option_value})
    assert result is expected