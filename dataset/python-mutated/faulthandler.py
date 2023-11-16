import os
import sys
from typing import Generator
import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from _pytest.stash import StashKey
fault_handler_stderr_fd_key = StashKey[int]()
fault_handler_originally_enabled_key = StashKey[bool]()

def pytest_addoption(parser: Parser) -> None:
    if False:
        print('Hello World!')
    help = 'Dump the traceback of all threads if a test takes more than TIMEOUT seconds to finish'
    parser.addini('faulthandler_timeout', help, default=0.0)

def pytest_configure(config: Config) -> None:
    if False:
        for i in range(10):
            print('nop')
    import faulthandler
    config.stash[fault_handler_stderr_fd_key] = os.dup(get_stderr_fileno())
    config.stash[fault_handler_originally_enabled_key] = faulthandler.is_enabled()
    faulthandler.enable(file=config.stash[fault_handler_stderr_fd_key])

def pytest_unconfigure(config: Config) -> None:
    if False:
        return 10
    import faulthandler
    faulthandler.disable()
    if fault_handler_stderr_fd_key in config.stash:
        os.close(config.stash[fault_handler_stderr_fd_key])
        del config.stash[fault_handler_stderr_fd_key]
    if config.stash.get(fault_handler_originally_enabled_key, False):
        faulthandler.enable(file=get_stderr_fileno())

def get_stderr_fileno() -> int:
    if False:
        return 10
    try:
        fileno = sys.stderr.fileno()
        if fileno == -1:
            raise AttributeError()
        return fileno
    except (AttributeError, ValueError):
        return sys.__stderr__.fileno()

def get_timeout_config_value(config: Config) -> float:
    if False:
        return 10
    return float(config.getini('faulthandler_timeout') or 0.0)

@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_protocol(item: Item) -> Generator[None, object, object]:
    if False:
        return 10
    timeout = get_timeout_config_value(item.config)
    if timeout > 0:
        import faulthandler
        stderr = item.config.stash[fault_handler_stderr_fd_key]
        faulthandler.dump_traceback_later(timeout, file=stderr)
        try:
            return (yield)
        finally:
            faulthandler.cancel_dump_traceback_later()
    else:
        return (yield)

@pytest.hookimpl(tryfirst=True)
def pytest_enter_pdb() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Cancel any traceback dumping due to timeout before entering pdb.'
    import faulthandler
    faulthandler.cancel_dump_traceback_later()

@pytest.hookimpl(tryfirst=True)
def pytest_exception_interact() -> None:
    if False:
        i = 10
        return i + 15
    'Cancel any traceback dumping due to an interactive exception being\n    raised.'
    import faulthandler
    faulthandler.cancel_dump_traceback_later()