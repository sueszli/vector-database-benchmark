"""Support for providing temporary directories to test functions."""
import dataclasses
import os
import re
import tempfile
from pathlib import Path
from shutil import rmtree
from typing import Any
from typing import Dict
from typing import final
from typing import Generator
from typing import Literal
from typing import Optional
from typing import Union
from .pathlib import cleanup_dead_symlinks
from .pathlib import LOCK_TIMEOUT
from .pathlib import make_numbered_dir
from .pathlib import make_numbered_dir_with_cleanup
from .pathlib import rm_rf
from _pytest.compat import get_user_id
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Item
from _pytest.reports import TestReport
from _pytest.stash import StashKey
tmppath_result_key = StashKey[Dict[str, bool]]()
RetentionType = Literal['all', 'failed', 'none']

@final
@dataclasses.dataclass
class TempPathFactory:
    """Factory for temporary directories under the common base temp directory.

    The base directory can be configured using the ``--basetemp`` option.
    """
    _given_basetemp: Optional[Path]
    _trace: Any
    _basetemp: Optional[Path]
    _retention_count: int
    _retention_policy: RetentionType

    def __init__(self, given_basetemp: Optional[Path], retention_count: int, retention_policy: RetentionType, trace, basetemp: Optional[Path]=None, *, _ispytest: bool=False) -> None:
        if False:
            return 10
        check_ispytest(_ispytest)
        if given_basetemp is None:
            self._given_basetemp = None
        else:
            self._given_basetemp = Path(os.path.abspath(str(given_basetemp)))
        self._trace = trace
        self._retention_count = retention_count
        self._retention_policy = retention_policy
        self._basetemp = basetemp

    @classmethod
    def from_config(cls, config: Config, *, _ispytest: bool=False) -> 'TempPathFactory':
        if False:
            for i in range(10):
                print('nop')
        'Create a factory according to pytest configuration.\n\n        :meta private:\n        '
        check_ispytest(_ispytest)
        count = int(config.getini('tmp_path_retention_count'))
        if count < 0:
            raise ValueError(f'tmp_path_retention_count must be >= 0. Current input: {count}.')
        policy = config.getini('tmp_path_retention_policy')
        if policy not in ('all', 'failed', 'none'):
            raise ValueError(f'tmp_path_retention_policy must be either all, failed, none. Current input: {policy}.')
        return cls(given_basetemp=config.option.basetemp, trace=config.trace.get('tmpdir'), retention_count=count, retention_policy=policy, _ispytest=True)

    def _ensure_relative_to_basetemp(self, basename: str) -> str:
        if False:
            return 10
        basename = os.path.normpath(basename)
        if (self.getbasetemp() / basename).resolve().parent != self.getbasetemp():
            raise ValueError(f'{basename} is not a normalized and relative path')
        return basename

    def mktemp(self, basename: str, numbered: bool=True) -> Path:
        if False:
            return 10
        'Create a new temporary directory managed by the factory.\n\n        :param basename:\n            Directory base name, must be a relative path.\n\n        :param numbered:\n            If ``True``, ensure the directory is unique by adding a numbered\n            suffix greater than any existing one: ``basename="foo-"`` and ``numbered=True``\n            means that this function will create directories named ``"foo-0"``,\n            ``"foo-1"``, ``"foo-2"`` and so on.\n\n        :returns:\n            The path to the new directory.\n        '
        basename = self._ensure_relative_to_basetemp(basename)
        if not numbered:
            p = self.getbasetemp().joinpath(basename)
            p.mkdir(mode=448)
        else:
            p = make_numbered_dir(root=self.getbasetemp(), prefix=basename, mode=448)
            self._trace('mktemp', p)
        return p

    def getbasetemp(self) -> Path:
        if False:
            i = 10
            return i + 15
        'Return the base temporary directory, creating it if needed.\n\n        :returns:\n            The base temporary directory.\n        '
        if self._basetemp is not None:
            return self._basetemp
        if self._given_basetemp is not None:
            basetemp = self._given_basetemp
            if basetemp.exists():
                rm_rf(basetemp)
            basetemp.mkdir(mode=448)
            basetemp = basetemp.resolve()
        else:
            from_env = os.environ.get('PYTEST_DEBUG_TEMPROOT')
            temproot = Path(from_env or tempfile.gettempdir()).resolve()
            user = get_user() or 'unknown'
            rootdir = temproot.joinpath(f'pytest-of-{user}')
            try:
                rootdir.mkdir(mode=448, exist_ok=True)
            except OSError:
                rootdir = temproot.joinpath('pytest-of-unknown')
                rootdir.mkdir(mode=448, exist_ok=True)
            uid = get_user_id()
            if uid is not None:
                rootdir_stat = rootdir.stat()
                if rootdir_stat.st_uid != uid:
                    raise OSError(f'The temporary directory {rootdir} is not owned by the current user. Fix this and try again.')
                if rootdir_stat.st_mode & 63 != 0:
                    os.chmod(rootdir, rootdir_stat.st_mode & ~63)
            keep = self._retention_count
            if self._retention_policy == 'none':
                keep = 0
            basetemp = make_numbered_dir_with_cleanup(prefix='pytest-', root=rootdir, keep=keep, lock_timeout=LOCK_TIMEOUT, mode=448)
        assert basetemp is not None, basetemp
        self._basetemp = basetemp
        self._trace('new basetemp', basetemp)
        return basetemp

def get_user() -> Optional[str]:
    if False:
        return 10
    'Return the current user name, or None if getuser() does not work\n    in the current environment (see #1010).'
    try:
        import getpass
        return getpass.getuser()
    except (ImportError, KeyError):
        return None

def pytest_configure(config: Config) -> None:
    if False:
        return 10
    'Create a TempPathFactory and attach it to the config object.\n\n    This is to comply with existing plugins which expect the handler to be\n    available at pytest_configure time, but ideally should be moved entirely\n    to the tmp_path_factory session fixture.\n    '
    mp = MonkeyPatch()
    config.add_cleanup(mp.undo)
    _tmp_path_factory = TempPathFactory.from_config(config, _ispytest=True)
    mp.setattr(config, '_tmp_path_factory', _tmp_path_factory, raising=False)

def pytest_addoption(parser: Parser) -> None:
    if False:
        while True:
            i = 10
    parser.addini('tmp_path_retention_count', help='How many sessions should we keep the `tmp_path` directories, according to `tmp_path_retention_policy`.', default=3)
    parser.addini('tmp_path_retention_policy', help='Controls which directories created by the `tmp_path` fixture are kept around, based on test outcome. (all/failed/none)', default='all')

@fixture(scope='session')
def tmp_path_factory(request: FixtureRequest) -> TempPathFactory:
    if False:
        while True:
            i = 10
    'Return a :class:`pytest.TempPathFactory` instance for the test session.'
    return request.config._tmp_path_factory

def _mk_tmp(request: FixtureRequest, factory: TempPathFactory) -> Path:
    if False:
        print('Hello World!')
    name = request.node.name
    name = re.sub('[\\W]', '_', name)
    MAXVAL = 30
    name = name[:MAXVAL]
    return factory.mktemp(name, numbered=True)

@fixture
def tmp_path(request: FixtureRequest, tmp_path_factory: TempPathFactory) -> Generator[Path, None, None]:
    if False:
        i = 10
        return i + 15
    'Return a temporary directory path object which is unique to each test\n    function invocation, created as a sub directory of the base temporary\n    directory.\n\n    By default, a new base temporary directory is created each test session,\n    and old bases are removed after 3 sessions, to aid in debugging.\n    This behavior can be configured with :confval:`tmp_path_retention_count` and\n    :confval:`tmp_path_retention_policy`.\n    If ``--basetemp`` is used then it is cleared each session. See :ref:`base\n    temporary directory`.\n\n    The returned object is a :class:`pathlib.Path` object.\n    '
    path = _mk_tmp(request, tmp_path_factory)
    yield path
    tmp_path_factory: TempPathFactory = request.session.config._tmp_path_factory
    policy = tmp_path_factory._retention_policy
    result_dict = request.node.stash[tmppath_result_key]
    if policy == 'failed' and result_dict.get('call', True):
        rmtree(path, ignore_errors=True)
    del request.node.stash[tmppath_result_key]

def pytest_sessionfinish(session, exitstatus: Union[int, ExitCode]):
    if False:
        return 10
    'After each session, remove base directory if all the tests passed,\n    the policy is "failed", and the basetemp is not specified by a user.\n    '
    tmp_path_factory: TempPathFactory = session.config._tmp_path_factory
    basetemp = tmp_path_factory._basetemp
    if basetemp is None:
        return
    policy = tmp_path_factory._retention_policy
    if exitstatus == 0 and policy == 'failed' and (tmp_path_factory._given_basetemp is None):
        if basetemp.is_dir():
            rmtree(basetemp, ignore_errors=True)
    if basetemp.is_dir():
        cleanup_dead_symlinks(basetemp)

@hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: Item, call) -> Generator[None, TestReport, TestReport]:
    if False:
        print('Hello World!')
    rep = (yield)
    assert rep.when is not None
    empty: Dict[str, bool] = {}
    item.stash.setdefault(tmppath_result_key, empty)[rep.when] = rep.passed
    return rep