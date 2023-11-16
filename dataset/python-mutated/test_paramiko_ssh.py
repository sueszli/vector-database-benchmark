from __future__ import annotations
from io import StringIO
import pytest
from ansible.plugins.connection import paramiko_ssh as paramiko_ssh_module
from ansible.plugins.loader import connection_loader
from ansible.playbook.play_context import PlayContext

@pytest.fixture
def play_context():
    if False:
        print('Hello World!')
    play_context = PlayContext()
    play_context.prompt = '[sudo via ansible, key=ouzmdnewuhucvuaabtjmweasarviygqq] password: '
    return play_context

@pytest.fixture()
def in_stream():
    if False:
        while True:
            i = 10
    return StringIO()

def test_paramiko_connection_module(play_context, in_stream):
    if False:
        return 10
    assert isinstance(connection_loader.get('paramiko_ssh', play_context, in_stream), paramiko_ssh_module.Connection)

def test_paramiko_connect(play_context, in_stream, mocker):
    if False:
        print('Hello World!')
    paramiko_ssh = connection_loader.get('paramiko_ssh', play_context, in_stream)
    mocker.patch.object(paramiko_ssh, '_connect_uncached')
    connection = paramiko_ssh._connect()
    assert isinstance(connection, paramiko_ssh_module.Connection)
    assert connection._connected is True