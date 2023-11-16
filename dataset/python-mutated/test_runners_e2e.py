"""Tests for runners."""
import logging
import pytest

def command_expansion_base(quteproc, send_msg, recv_msg, url='data/hello.txt'):
    if False:
        for i in range(10):
            print('nop')
    quteproc.open_path(url)
    quteproc.send_cmd(':message-info ' + send_msg)
    quteproc.mark_expected(category='message', loglevel=logging.INFO, message=recv_msg)

@pytest.mark.parametrize('send_msg, recv_msg', [('foo{{url}}bar', 'foo{url}bar'), ('foo{url}', 'foohttp://localhost:*/hello.txt'), ('foo{url:pretty}', 'foohttp://localhost:*/hello.txt'), ('foo{url:domain}', 'foohttp://localhost:*'), ('foo{url:auth}', 'foo'), ('foo{url:scheme}', 'foohttp'), ('foo{url:host}', 'foolocalhost'), ('foo{url:path}', 'foo*/hello.txt')])
def test_command_expansion(quteproc, send_msg, recv_msg):
    if False:
        return 10
    command_expansion_base(quteproc, send_msg, recv_msg)

@pytest.mark.parametrize('send_msg, recv_msg, url', [('foo{title}', 'fooTest title', 'data/title.html'), ('foo{url:query}', 'fooq=bar', 'data/hello.txt?q=bar'), ('{title}bar{url}', 'Test titlebarhttp://localhost:*/title.html', 'data/title.html')])
def test_command_expansion_complex(quteproc, send_msg, recv_msg, url):
    if False:
        return 10
    command_expansion_base(quteproc, send_msg, recv_msg, url)

def test_command_expansion_basic_auth(quteproc, server):
    if False:
        return 10
    url = 'http://user1:password1@localhost:{port}/basic-auth/user1/password1'.format(port=server.port)
    quteproc.open_url(url)
    quteproc.send_cmd(':message-info foo{url:auth}')
    quteproc.mark_expected(category='message', loglevel=logging.INFO, message='foouser1:password1@')

def test_command_expansion_clipboard(quteproc):
    if False:
        return 10
    quteproc.send_cmd(':debug-set-fake-clipboard "foo"')
    command_expansion_base(quteproc, '{clipboard}bar{url}', 'foobarhttp://localhost:*/hello.txt')
    quteproc.send_cmd(':debug-set-fake-clipboard "{{url}}"')
    command_expansion_base(quteproc, '{clipboard}bar{url}', '{url}barhttp://localhost:*/hello.txt')