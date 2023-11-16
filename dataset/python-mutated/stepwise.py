from typing import List
from typing import Optional
from typing import TYPE_CHECKING
import pytest
from _pytest import nodes
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.main import Session
from _pytest.reports import TestReport
if TYPE_CHECKING:
    from _pytest.cacheprovider import Cache
STEPWISE_CACHE_DIR = 'cache/stepwise'

def pytest_addoption(parser: Parser) -> None:
    if False:
        for i in range(10):
            print('nop')
    group = parser.getgroup('general')
    group.addoption('--sw', '--stepwise', action='store_true', default=False, dest='stepwise', help='Exit on test failure and continue from last failing test next time')
    group.addoption('--sw-skip', '--stepwise-skip', action='store_true', default=False, dest='stepwise_skip', help='Ignore the first failing test but stop on the next failing test. Implicitly enables --stepwise.')

@pytest.hookimpl
def pytest_configure(config: Config) -> None:
    if False:
        print('Hello World!')
    if config.option.stepwise_skip:
        config.option.stepwise = True
    if config.getoption('stepwise'):
        config.pluginmanager.register(StepwisePlugin(config), 'stepwiseplugin')

def pytest_sessionfinish(session: Session) -> None:
    if False:
        print('Hello World!')
    if not session.config.getoption('stepwise'):
        assert session.config.cache is not None
        if hasattr(session.config, 'workerinput'):
            return
        session.config.cache.set(STEPWISE_CACHE_DIR, [])

class StepwisePlugin:

    def __init__(self, config: Config) -> None:
        if False:
            return 10
        self.config = config
        self.session: Optional[Session] = None
        self.report_status = ''
        assert config.cache is not None
        self.cache: Cache = config.cache
        self.lastfailed: Optional[str] = self.cache.get(STEPWISE_CACHE_DIR, None)
        self.skip: bool = config.getoption('stepwise_skip')

    def pytest_sessionstart(self, session: Session) -> None:
        if False:
            i = 10
            return i + 15
        self.session = session

    def pytest_collection_modifyitems(self, config: Config, items: List[nodes.Item]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not self.lastfailed:
            self.report_status = 'no previously failed tests, not skipping.'
            return
        failed_index = None
        for (index, item) in enumerate(items):
            if item.nodeid == self.lastfailed:
                failed_index = index
                break
        if failed_index is None:
            self.report_status = 'previously failed test not found, not skipping.'
        else:
            self.report_status = f'skipping {failed_index} already passed items.'
            deselected = items[:failed_index]
            del items[:failed_index]
            config.hook.pytest_deselected(items=deselected)

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        if False:
            print('Hello World!')
        if report.failed:
            if self.skip:
                if report.nodeid == self.lastfailed:
                    self.lastfailed = None
                self.skip = False
            else:
                self.lastfailed = report.nodeid
                assert self.session is not None
                self.session.shouldstop = 'Test failed, continuing from this test next run.'
        elif report.when == 'call':
            if report.nodeid == self.lastfailed:
                self.lastfailed = None

    def pytest_report_collectionfinish(self) -> Optional[str]:
        if False:
            return 10
        if self.config.getoption('verbose') >= 0 and self.report_status:
            return f'stepwise: {self.report_status}'
        return None

    def pytest_sessionfinish(self) -> None:
        if False:
            i = 10
            return i + 15
        if hasattr(self.config, 'workerinput'):
            return
        self.cache.set(STEPWISE_CACHE_DIR, self.lastfailed)