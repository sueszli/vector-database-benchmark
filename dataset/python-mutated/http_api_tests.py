"""
Tests that emulate the debugger adaptor and just test the interaction between
the front end and back end API classes. HTTP edition!

Tests:
Server (via HTTP)
"""
import logging
import sys
import json
import time
import subprocess
from nose.tools import *
import voltron
from voltron.core import *
from voltron.api import *
from voltron.plugin import *
import platform
if platform.system() == 'Darwin':
    sys.path.append('/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Resources/Python')
from .common import *
import requests
log = logging.getLogger('tests')

class APIHostNotSupportedRequest(APIRequest):

    @server_side
    def dispatch(self):
        if False:
            return 10
        return APIDebuggerHostNotSupportedErrorResponse()

class APIHostNotSupportedPlugin(APIPlugin):
    request = 'host_not_supported'
    request_class = APIHostNotSupportedRequest
    response_class = APIResponse

def setup():
    if False:
        while True:
            i = 10
    global server, client, target, pm, adaptor, methods
    log.info('setting up API tests')
    voltron.setup_env()
    voltron.config['server'] = {'listen': {'tcp': ['127.0.0.1', 5555]}}
    pm = PluginManager()
    plugin = pm.debugger_plugin_for_host('mock')
    adaptor = plugin.adaptor_class()
    voltron.debugger = adaptor
    inject_mock(adaptor)
    server = Server()
    server.start()
    time.sleep(2)

def teardown():
    if False:
        i = 10
        return i + 15
    server.stop()
    time.sleep(2)

def test_disassemble():
    if False:
        print('Hello World!')
    data = requests.get('http://localhost:5555/api/disassemble?count=16').text
    res = APIResponse(data=data)
    assert res.is_success
    assert res.disassembly == disassemble_response

def test_command():
    if False:
        i = 10
        return i + 15
    data = requests.get('http://localhost:5555/api/command?command=reg%20read').text
    res = APIResponse(data=data)
    assert res.is_success
    assert res.output == command_response

def test_targets():
    if False:
        return 10
    data = requests.get('http://localhost:5555/api/targets').text
    res = api_response('targets', data=data)
    assert res.is_success
    assert res.targets == targets_response

def test_memory():
    if False:
        for i in range(10):
            print('nop')
    data = requests.get('http://localhost:5555/api/registers').text
    res = api_response('registers', data=data)
    url = 'http://localhost:5555/api/memory?address={}&length=64'.format(res.registers['rip'])
    data = requests.get(url).text
    res = api_response('memory', data=data)
    assert res.is_success
    assert res.memory == memory_response

def test_registers():
    if False:
        i = 10
        return i + 15
    data = requests.get('http://localhost:5555/api/registers').text
    res = api_response('registers', data=data)
    assert res.is_success
    assert res.registers == registers_response

def test_stack_length_missing():
    if False:
        while True:
            i = 10
    data = requests.get('http://localhost:5555/api/stack').text
    res = APIErrorResponse(data=data)
    assert res.is_error
    assert res.message == 'length'

def test_stack():
    if False:
        return 10
    data = requests.get('http://localhost:5555/api/stack?length=64').text
    res = api_response('stack', data=data)
    assert res.is_success
    assert res.memory == stack_response

def test_state():
    if False:
        print('Hello World!')
    data = requests.get('http://localhost:5555/api/state').text
    res = api_response('state', data=data)
    assert res.is_success
    assert res.state == state_response

def test_version():
    if False:
        print('Hello World!')
    data = requests.get('http://localhost:5555/api/version').text
    res = api_response('version', data=data)
    assert res.is_success
    assert res.api_version == 1.1
    assert res.host_version == 'lldb-something'

def test_bad_json():
    if False:
        return 10
    data = requests.post('http://localhost:5555/api/request', data='xxx').text
    res = APIResponse(data=data)
    assert res.is_error
    assert res.code == 4097

def test_bad_request():
    if False:
        i = 10
        return i + 15
    data = requests.post('http://localhost:5555/api/request', data='{"type":"request","request":"no_such_request"}').text
    res = APIResponse(data=data)
    assert res.is_error
    assert res.code == 4098

def test_breakpoints():
    if False:
        while True:
            i = 10
    data = requests.get('http://localhost:5555/api/breakpoints').text
    res = api_response('breakpoints', data=data)
    assert res.is_success
    assert res.breakpoints == breakpoints_response