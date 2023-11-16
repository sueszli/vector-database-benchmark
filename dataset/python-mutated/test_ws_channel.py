import atexit
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
from subprocess import Popen, PIPE
import sys
import time
from nni.runtime.command_channel.websocket import WsChannelClient
_server = None
_client = None
_command1 = {'type': 'ut_command', 'value': 123}
_command2 = {'type': 'ut_command', 'value': '你好'}

def test_connect():
    if False:
        while True:
            i = 10
    global _client
    port = _init()
    _client = WsChannelClient(f'ws://localhost:{port}')
    _client.connect()

def test_send():
    if False:
        while True:
            i = 10
    _client.send(_command1)
    _client.send(_command2)
    time.sleep(0.01)
    sent1 = json.loads(_server.stdout.readline())
    assert sent1 == _command1, sent1
    sent2 = json.loads(_server.stdout.readline().strip())
    assert sent2 == _command2, sent2

def test_receive():
    if False:
        return 10
    _server.stdin.write(json.dumps(_command1) + '\n')
    _server.stdin.write(json.dumps(_command2) + '\n')
    _server.stdin.flush()
    received1 = _client.receive()
    assert received1 == _command1, received1
    received2 = _client.receive()
    assert received2 == _command2, received2

def test_disconnect():
    if False:
        for i in range(10):
            print('nop')
    _client.disconnect()
    global _server
    _server.stdin.write('_close_\n')
    _server.stdin.flush()
    time.sleep(0.1)
    _server.terminate()
    _server = None

def _init():
    if False:
        print('Hello World!')
    global _server
    script = (Path(__file__).parent / 'helper/websocket_server.py').resolve()
    _server = Popen([sys.executable, str(script)], stdin=PIPE, stdout=PIPE, encoding='utf_8')
    time.sleep(0.1)
    atexit.register(lambda : _server is None or _server.terminate())
    return int(_server.stdout.readline().strip())
if __name__ == '__main__':
    test_connect()
    test_send()
    test_receive()
    test_disconnect()
    print('pass')