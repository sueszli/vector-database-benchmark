"""Submit failure or test session information to a pastebin service."""
import tempfile
from io import StringIO
from typing import IO
from typing import Union
import pytest
from _pytest.config import Config
from _pytest.config import create_terminal_writer
from _pytest.config.argparsing import Parser
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
pastebinfile_key = StashKey[IO[bytes]]()

def pytest_addoption(parser: Parser) -> None:
    if False:
        print('Hello World!')
    group = parser.getgroup('terminal reporting')
    group._addoption('--pastebin', metavar='mode', action='store', dest='pastebin', default=None, choices=['failed', 'all'], help='Send failed|all info to bpaste.net pastebin service')

@pytest.hookimpl(trylast=True)
def pytest_configure(config: Config) -> None:
    if False:
        for i in range(10):
            print('nop')
    if config.option.pastebin == 'all':
        tr = config.pluginmanager.getplugin('terminalreporter')
        if tr is not None:
            config.stash[pastebinfile_key] = tempfile.TemporaryFile('w+b')
            oldwrite = tr._tw.write

            def tee_write(s, **kwargs):
                if False:
                    return 10
                oldwrite(s, **kwargs)
                if isinstance(s, str):
                    s = s.encode('utf-8')
                config.stash[pastebinfile_key].write(s)
            tr._tw.write = tee_write

def pytest_unconfigure(config: Config) -> None:
    if False:
        print('Hello World!')
    if pastebinfile_key in config.stash:
        pastebinfile = config.stash[pastebinfile_key]
        pastebinfile.seek(0)
        sessionlog = pastebinfile.read()
        pastebinfile.close()
        del config.stash[pastebinfile_key]
        tr = config.pluginmanager.getplugin('terminalreporter')
        del tr._tw.__dict__['write']
        tr.write_sep('=', 'Sending information to Paste Service')
        pastebinurl = create_new_paste(sessionlog)
        tr.write_line('pastebin session-log: %s\n' % pastebinurl)

def create_new_paste(contents: Union[str, bytes]) -> str:
    if False:
        i = 10
        return i + 15
    'Create a new paste using the bpaste.net service.\n\n    :contents: Paste contents string.\n    :returns: URL to the pasted contents, or an error message.\n    '
    import re
    from urllib.request import urlopen
    from urllib.parse import urlencode
    params = {'code': contents, 'lexer': 'text', 'expiry': '1week'}
    url = 'https://bpa.st'
    try:
        response: str = urlopen(url, data=urlencode(params).encode('ascii')).read().decode('utf-8')
    except OSError as exc_info:
        return 'bad response: %s' % exc_info
    m = re.search('href="/raw/(\\w+)"', response)
    if m:
        return f'{url}/show/{m.group(1)}'
    else:
        return "bad response: invalid format ('" + response + "')"

def pytest_terminal_summary(terminalreporter: TerminalReporter) -> None:
    if False:
        for i in range(10):
            print('nop')
    if terminalreporter.config.option.pastebin != 'failed':
        return
    if 'failed' in terminalreporter.stats:
        terminalreporter.write_sep('=', 'Sending information to Paste Service')
        for rep in terminalreporter.stats['failed']:
            try:
                msg = rep.longrepr.reprtraceback.reprentries[-1].reprfileloc
            except AttributeError:
                msg = terminalreporter._getfailureheadline(rep)
            file = StringIO()
            tw = create_terminal_writer(terminalreporter.config, file)
            rep.toterminal(tw)
            s = file.getvalue()
            assert len(s)
            pastebinurl = create_new_paste(s)
            terminalreporter.write_line(f'{msg} --> {pastebinurl}')