"""
Tests that test voltron in the gdb cli driver

Tests:
Client -> Server -> GDBAdaptor

Inside a GDB instance
"""
from __future__ import print_function
import tempfile
import sys
import json
import time
import logging
import pexpect
import os
import six
from mock import Mock
from nose.tools import *
import voltron
from voltron.core import *
from voltron.api import *
from voltron.plugin import PluginManager, DebuggerAdaptorPlugin
from .common import *
log = logging.getLogger('tests')
p = None
client = None

def setup():
    if False:
        i = 10
        return i + 15
    global p, client, pm
    log.info('setting up GDB CLI tests')
    voltron.setup_env()
    pexpect.run('cc -o tests/inferior tests/inferior.c')
    start_debugger()

def teardown():
    if False:
        i = 10
        return i + 15
    read_data()
    p.terminate(True)

def start_debugger(do_break=True):
    if False:
        return 10
    global p, client
    p = pexpect.spawn('gdb')
    p.sendline("python import sys;sys.path.append('/home/travis/virtualenv/python3.5.0/lib/python3.5/site-packages')")
    p.sendline("python import sys;sys.path.append('/home/travis/virtualenv/python3.4.3/lib/python3.4/site-packages')")
    p.sendline("python import sys;sys.path.append('/home/travis/virtualenv/python3.3.6/lib/python3.3/site-packages')")
    p.sendline("python import sys;sys.path.append('/home/travis/virtualenv/python2.7.10/lib/python2.7/site-packages')")
    p.sendline('source voltron/entry.py')
    p.sendline('file tests/inferior')
    p.sendline('set disassembly-flavor intel')
    p.sendline('voltron init')
    if do_break:
        p.sendline('b main')
    p.sendline('run loop')
    read_data()
    time.sleep(5)
    client = Client()

def stop_debugger():
    if False:
        i = 10
        return i + 15
    read_data()
    p.terminate(True)

def read_data():
    if False:
        return 10
    try:
        while True:
            data = p.read_nonblocking(size=64, timeout=1)
            print(data.decode('UTF-8'), end='')
    except:
        pass

def restart_debugger(do_break=True):
    if False:
        return 10
    stop_debugger()
    start_debugger(do_break)

def test_bad_request():
    if False:
        while True:
            i = 10
    req = client.create_request('version')
    req.request = 'xxx'
    res = client.send_request(req)
    assert res.is_error
    assert res.code == 4098

def test_version():
    if False:
        return 10
    req = client.create_request('version')
    res = client.send_request(req)
    assert res.api_version == 1.1
    assert 'gdb' in res.host_version

def test_registers():
    if False:
        i = 10
        return i + 15
    global registers
    read_data()
    res = client.perform_request('registers')
    registers = res.registers
    assert res.status == 'success'
    assert len(registers) > 0
    assert registers['rip'] != 0

def test_memory():
    if False:
        for i in range(10):
            print('nop')
    res = client.perform_request('memory', address=registers['rip'], length=64)
    assert res.status == 'success'
    assert len(res.memory) > 0

def test_state_stopped():
    if False:
        return 10
    res = client.perform_request('state')
    assert res.is_success
    assert res.state == 'stopped'

def test_targets():
    if False:
        return 10
    res = client.perform_request('targets')
    assert res.is_success
    assert res.targets[0]['state'] == 'stopped'
    assert res.targets[0]['arch'] == 'x86_64'
    assert res.targets[0]['id'] == 0
    assert res.targets[0]['file'].endswith('tests/inferior')

def test_stack():
    if False:
        print('Hello World!')
    res = client.perform_request('stack', length=64)
    assert res.status == 'success'
    assert len(res.memory) > 0

def test_command():
    if False:
        i = 10
        return i + 15
    res = client.perform_request('command', command='info reg')
    assert res.status == 'success'
    assert len(res.output) > 0
    assert 'rax' in res.output

def test_disassemble():
    if False:
        i = 10
        return i + 15
    res = client.perform_request('disassemble', count=32)
    assert res.status == 'success'
    assert len(res.disassembly) > 0
    assert 'DWORD' in res.disassembly

def test_backtrace():
    if False:
        for i in range(10):
            print('nop')
    res = client.perform_request('backtrace')
    print(res)
    assert res.is_success
    assert res.frames[0]['name'] == 'main'
    assert res.frames[0]['index'] == 0