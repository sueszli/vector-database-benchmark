from unittest.mock import Mock
import main

def test_print_name():
    if False:
        print('Hello World!')
    name = 'test'
    data = {'name': name}
    req = Mock(get_json=Mock(return_value=data), args=data)
    assert main.hello_http(req) == f'Hello {name}!'

def test_print_hello_world():
    if False:
        i = 10
        return i + 15
    data = {}
    req = Mock(get_json=Mock(return_value=data), args=data)
    assert main.hello_http(req) == 'Hello World!'