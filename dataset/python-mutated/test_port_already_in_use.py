from io import BytesIO
import pytest
from thefuck.rules.port_already_in_use import match, get_new_command
from thefuck.types import Command
outputs = ["\n\nDE 70% 1/1 build modulesevents.js:141\n      throw er; // Unhandled 'error' event\n      ^\n\nError: listen EADDRINUSE 127.0.0.1:8080\n    at Object.exports._errnoException (util.js:873:11)\n    at exports._exceptionWithHostPort (util.js:896:20)\n    at Server._listen2 (net.js:1250:14)\n    at listen (net.js:1286:10)\n    at net.js:1395:9\n    at GetAddrInfoReqWrap.asyncCallback [as callback] (dns.js:64:16)\n    at GetAddrInfoReqWrap.onlookup [as oncomplete] (dns.js:83:10)\n\n    ", "\n[6:40:01 AM] <START> Building Dependency Graph\n[6:40:01 AM] <START> Crawling File System\n ERROR  Packager can't listen on port 8080\nMost likely another process is already using this port\nRun the following command to find out which process:\n\n   lsof -n -i4TCP:8080\n\nYou can either shut down the other process:\n\n   kill -9 <PID>\n\nor run packager on different port.\n\n    ", '\nTraceback (most recent call last):\n  File "/usr/lib/python3.5/runpy.py", line 184, in _run_module_as_main\n    "__main__", mod_spec)\n  File "/usr/lib/python3.5/runpy.py", line 85, in _run_code\n    exec(code, run_globals)\n  File "/home/nvbn/exp/code_view/server/code_view/main.py", line 14, in <module>\n    web.run_app(app)\n  File "/home/nvbn/.virtualenvs/code_view/lib/python3.5/site-packages/aiohttp/web.py", line 310, in run_app\n    backlog=backlog))\n  File "/usr/lib/python3.5/asyncio/base_events.py", line 373, in run_until_complete\n    return future.result()\n  File "/usr/lib/python3.5/asyncio/futures.py", line 274, in result\n    raise self._exception\n  File "/usr/lib/python3.5/asyncio/tasks.py", line 240, in _step\n    result = coro.send(None)\n  File "/usr/lib/python3.5/asyncio/base_events.py", line 953, in create_server\n    % (sa, err.strerror.lower()))\nOSError: [Errno 98] error while attempting to bind on address (\'0.0.0.0\', 8080): address already in use\nTask was destroyed but it is pending!\ntask: <Task pending coro=<RedisProtocol._reader_coroutine() running at /home/nvbn/.virtualenvs/code_view/lib/python3.5/site-packages/asyncio_redis/protocol.py:921> wait_for=<Future pending cb=[Task._wakeup()]>>\n    ']
lsof_stdout = b'COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\nnode    18233 nvbn   16u  IPv4 557134      0t0  TCP localhost:http-alt (LISTEN)\n'

@pytest.fixture(autouse=True)
def lsof(mocker):
    if False:
        i = 10
        return i + 15
    patch = mocker.patch('thefuck.rules.port_already_in_use.Popen')
    patch.return_value.stdout = BytesIO(lsof_stdout)
    return patch

@pytest.mark.usefixtures('no_memoize')
@pytest.mark.parametrize('command', [Command('./app', output) for output in outputs] + [Command('./app', output) for output in outputs])
def test_match(command):
    if False:
        i = 10
        return i + 15
    assert match(command)

@pytest.mark.usefixtures('no_memoize')
@pytest.mark.parametrize('command, lsof_output', [(Command('./app', ''), lsof_stdout), (Command('./app', outputs[1]), b''), (Command('./app', outputs[2]), b'')])
def test_not_match(lsof, command, lsof_output):
    if False:
        while True:
            i = 10
    lsof.return_value.stdout = BytesIO(lsof_output)
    assert not match(command)

@pytest.mark.parametrize('command', [Command('./app', output) for output in outputs] + [Command('./app', output) for output in outputs])
def test_get_new_command(command):
    if False:
        return 10
    assert get_new_command(command) == 'kill 18233 && ./app'