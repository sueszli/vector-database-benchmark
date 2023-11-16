"""Hook specifications for pytest plugins which are invoked by pytest itself
and by builtin plugins."""
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from pluggy import HookspecMarker
from _pytest.deprecated import WARNING_CMDLINE_PREPARSE_HOOK
if TYPE_CHECKING:
    import pdb
    import warnings
    from typing import Literal
    from _pytest._code.code import ExceptionRepr
    from _pytest._code.code import ExceptionInfo
    from _pytest.config import Config
    from _pytest.config import ExitCode
    from _pytest.config import PytestPluginManager
    from _pytest.config import _PluggyPlugin
    from _pytest.config.argparsing import Parser
    from _pytest.fixtures import FixtureDef
    from _pytest.fixtures import SubRequest
    from _pytest.main import Session
    from _pytest.nodes import Collector
    from _pytest.nodes import Item
    from _pytest.outcomes import Exit
    from _pytest.python import Class
    from _pytest.python import Function
    from _pytest.python import Metafunc
    from _pytest.python import Module
    from _pytest.reports import CollectReport
    from _pytest.reports import TestReport
    from _pytest.runner import CallInfo
    from _pytest.terminal import TerminalReporter
    from _pytest.terminal import TestShortLogReport
    from _pytest.compat import LEGACY_PATH
hookspec = HookspecMarker('pytest')

@hookspec(historic=True)
def pytest_addhooks(pluginmanager: 'PytestPluginManager') -> None:
    if False:
        while True:
            i = 10
    'Called at plugin registration time to allow adding new hooks via a call to\n    ``pluginmanager.add_hookspecs(module_or_class, prefix)``.\n\n    :param pytest.PytestPluginManager pluginmanager: The pytest plugin manager.\n\n    .. note::\n        This hook is incompatible with hook wrappers.\n    '

@hookspec(historic=True)
def pytest_plugin_registered(plugin: '_PluggyPlugin', manager: 'PytestPluginManager') -> None:
    if False:
        for i in range(10):
            print('nop')
    'A new pytest plugin got registered.\n\n    :param plugin: The plugin module or instance.\n    :param pytest.PytestPluginManager manager: pytest plugin manager.\n\n    .. note::\n        This hook is incompatible with hook wrappers.\n    '

@hookspec(historic=True)
def pytest_addoption(parser: 'Parser', pluginmanager: 'PytestPluginManager') -> None:
    if False:
        return 10
    "Register argparse-style options and ini-style config values,\n    called once at the beginning of a test run.\n\n    .. note::\n\n        This function should be implemented only in plugins or ``conftest.py``\n        files situated at the tests root directory due to how pytest\n        :ref:`discovers plugins during startup <pluginorder>`.\n\n    :param pytest.Parser parser:\n        To add command line options, call\n        :py:func:`parser.addoption(...) <pytest.Parser.addoption>`.\n        To add ini-file values call :py:func:`parser.addini(...)\n        <pytest.Parser.addini>`.\n\n    :param pytest.PytestPluginManager pluginmanager:\n        The pytest plugin manager, which can be used to install :py:func:`hookspec`'s\n        or :py:func:`hookimpl`'s and allow one plugin to call another plugin's hooks\n        to change how command line options are added.\n\n    Options can later be accessed through the\n    :py:class:`config <pytest.Config>` object, respectively:\n\n    - :py:func:`config.getoption(name) <pytest.Config.getoption>` to\n      retrieve the value of a command line option.\n\n    - :py:func:`config.getini(name) <pytest.Config.getini>` to retrieve\n      a value read from an ini-style file.\n\n    The config object is passed around on many internal objects via the ``.config``\n    attribute or can be retrieved as the ``pytestconfig`` fixture.\n\n    .. note::\n        This hook is incompatible with hook wrappers.\n    "

@hookspec(historic=True)
def pytest_configure(config: 'Config') -> None:
    if False:
        while True:
            i = 10
    'Allow plugins and conftest files to perform initial configuration.\n\n    This hook is called for every plugin and initial conftest file\n    after command line options have been parsed.\n\n    After that, the hook is called for other conftest files as they are\n    imported.\n\n    .. note::\n        This hook is incompatible with hook wrappers.\n\n    :param pytest.Config config: The pytest config object.\n    '

@hookspec(firstresult=True)
def pytest_cmdline_parse(pluginmanager: 'PytestPluginManager', args: List[str]) -> Optional['Config']:
    if False:
        while True:
            i = 10
    'Return an initialized :class:`~pytest.Config`, parsing the specified args.\n\n    Stops at first non-None result, see :ref:`firstresult`.\n\n    .. note::\n        This hook will only be called for plugin classes passed to the\n        ``plugins`` arg when using `pytest.main`_ to perform an in-process\n        test run.\n\n    :param pluginmanager: The pytest plugin manager.\n    :param args: List of arguments passed on the command line.\n    :returns: A pytest config object.\n    '

@hookspec(warn_on_impl=WARNING_CMDLINE_PREPARSE_HOOK)
def pytest_cmdline_preparse(config: 'Config', args: List[str]) -> None:
    if False:
        while True:
            i = 10
    '(**Deprecated**) modify command line arguments before option parsing.\n\n    This hook is considered deprecated and will be removed in a future pytest version. Consider\n    using :hook:`pytest_load_initial_conftests` instead.\n\n    .. note::\n        This hook will not be called for ``conftest.py`` files, only for setuptools plugins.\n\n    :param config: The pytest config object.\n    :param args: Arguments passed on the command line.\n    '

@hookspec(firstresult=True)
def pytest_cmdline_main(config: 'Config') -> Optional[Union['ExitCode', int]]:
    if False:
        return 10
    'Called for performing the main command line action. The default\n    implementation will invoke the configure hooks and runtest_mainloop.\n\n    Stops at first non-None result, see :ref:`firstresult`.\n\n    :param config: The pytest config object.\n    :returns: The exit code.\n    '

def pytest_load_initial_conftests(early_config: 'Config', parser: 'Parser', args: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Called to implement the loading of initial conftest files ahead\n    of command line option parsing.\n\n    .. note::\n        This hook will not be called for ``conftest.py`` files, only for setuptools plugins.\n\n    :param early_config: The pytest config object.\n    :param args: Arguments passed on the command line.\n    :param parser: To add command line options.\n    '

@hookspec(firstresult=True)
def pytest_collection(session: 'Session') -> Optional[object]:
    if False:
        return 10
    'Perform the collection phase for the given session.\n\n    Stops at first non-None result, see :ref:`firstresult`.\n    The return value is not used, but only stops further processing.\n\n    The default collection phase is this (see individual hooks for full details):\n\n    1. Starting from ``session`` as the initial collector:\n\n      1. ``pytest_collectstart(collector)``\n      2. ``report = pytest_make_collect_report(collector)``\n      3. ``pytest_exception_interact(collector, call, report)`` if an interactive exception occurred\n      4. For each collected node:\n\n        1. If an item, ``pytest_itemcollected(item)``\n        2. If a collector, recurse into it.\n\n      5. ``pytest_collectreport(report)``\n\n    2. ``pytest_collection_modifyitems(session, config, items)``\n\n      1. ``pytest_deselected(items)`` for any deselected items (may be called multiple times)\n\n    3. ``pytest_collection_finish(session)``\n    4. Set ``session.items`` to the list of collected items\n    5. Set ``session.testscollected`` to the number of collected items\n\n    You can implement this hook to only perform some action before collection,\n    for example the terminal plugin uses it to start displaying the collection\n    counter (and returns `None`).\n\n    :param session: The pytest session object.\n    '

def pytest_collection_modifyitems(session: 'Session', config: 'Config', items: List['Item']) -> None:
    if False:
        i = 10
        return i + 15
    'Called after collection has been performed. May filter or re-order\n    the items in-place.\n\n    :param session: The pytest session object.\n    :param config: The pytest config object.\n    :param items: List of item objects.\n    '

def pytest_collection_finish(session: 'Session') -> None:
    if False:
        for i in range(10):
            print('nop')
    'Called after collection has been performed and modified.\n\n    :param session: The pytest session object.\n    '

@hookspec(firstresult=True)
def pytest_ignore_collect(collection_path: Path, path: 'LEGACY_PATH', config: 'Config') -> Optional[bool]:
    if False:
        while True:
            i = 10
    'Return True to prevent considering this path for collection.\n\n    This hook is consulted for all files and directories prior to calling\n    more specific hooks.\n\n    Stops at first non-None result, see :ref:`firstresult`.\n\n    :param collection_path: The path to analyze.\n    :param path: The path to analyze (deprecated).\n    :param config: The pytest config object.\n\n    .. versionchanged:: 7.0.0\n        The ``collection_path`` parameter was added as a :class:`pathlib.Path`\n        equivalent of the ``path`` parameter. The ``path`` parameter\n        has been deprecated.\n    '

def pytest_collect_file(file_path: Path, path: 'LEGACY_PATH', parent: 'Collector') -> 'Optional[Collector]':
    if False:
        return 10
    'Create a :class:`~pytest.Collector` for the given path, or None if not relevant.\n\n    The new node needs to have the specified ``parent`` as a parent.\n\n    :param file_path: The path to analyze.\n    :param path: The path to collect (deprecated).\n\n    .. versionchanged:: 7.0.0\n        The ``file_path`` parameter was added as a :class:`pathlib.Path`\n        equivalent of the ``path`` parameter. The ``path`` parameter\n        has been deprecated.\n    '

def pytest_collectstart(collector: 'Collector') -> None:
    if False:
        i = 10
        return i + 15
    'Collector starts collecting.\n\n    :param collector:\n        The collector.\n    '

def pytest_itemcollected(item: 'Item') -> None:
    if False:
        while True:
            i = 10
    'We just collected a test item.\n\n    :param item:\n        The item.\n    '

def pytest_collectreport(report: 'CollectReport') -> None:
    if False:
        return 10
    'Collector finished collecting.\n\n    :param report:\n        The collect report.\n    '

def pytest_deselected(items: Sequence['Item']) -> None:
    if False:
        while True:
            i = 10
    'Called for deselected test items, e.g. by keyword.\n\n    May be called multiple times.\n\n    :param items:\n        The items.\n    '

@hookspec(firstresult=True)
def pytest_make_collect_report(collector: 'Collector') -> 'Optional[CollectReport]':
    if False:
        while True:
            i = 10
    'Perform :func:`collector.collect() <pytest.Collector.collect>` and return\n    a :class:`~pytest.CollectReport`.\n\n    Stops at first non-None result, see :ref:`firstresult`.\n\n    :param collector:\n        The collector.\n    '

@hookspec(firstresult=True)
def pytest_pycollect_makemodule(module_path: Path, path: 'LEGACY_PATH', parent) -> Optional['Module']:
    if False:
        i = 10
        return i + 15
    'Return a :class:`pytest.Module` collector or None for the given path.\n\n    This hook will be called for each matching test module path.\n    The :hook:`pytest_collect_file` hook needs to be used if you want to\n    create test modules for files that do not match as a test module.\n\n    Stops at first non-None result, see :ref:`firstresult`.\n\n    :param module_path: The path of the module to collect.\n    :param path: The path of the module to collect (deprecated).\n\n    .. versionchanged:: 7.0.0\n        The ``module_path`` parameter was added as a :class:`pathlib.Path`\n        equivalent of the ``path`` parameter.\n\n        The ``path`` parameter has been deprecated in favor of ``fspath``.\n    '

@hookspec(firstresult=True)
def pytest_pycollect_makeitem(collector: Union['Module', 'Class'], name: str, obj: object) -> Union[None, 'Item', 'Collector', List[Union['Item', 'Collector']]]:
    if False:
        i = 10
        return i + 15
    'Return a custom item/collector for a Python object in a module, or None.\n\n    Stops at first non-None result, see :ref:`firstresult`.\n\n    :param collector:\n        The module/class collector.\n    :param name:\n        The name of the object in the module/class.\n    :param obj:\n        The object.\n    :returns:\n        The created items/collectors.\n    '

@hookspec(firstresult=True)
def pytest_pyfunc_call(pyfuncitem: 'Function') -> Optional[object]:
    if False:
        print('Hello World!')
    'Call underlying test function.\n\n    Stops at first non-None result, see :ref:`firstresult`.\n\n    :param pyfuncitem:\n        The function item.\n    '

def pytest_generate_tests(metafunc: 'Metafunc') -> None:
    if False:
        for i in range(10):
            print('nop')
    'Generate (multiple) parametrized calls to a test function.\n\n    :param metafunc:\n        The :class:`~pytest.Metafunc` helper for the test function.\n    '

@hookspec(firstresult=True)
def pytest_make_parametrize_id(config: 'Config', val: object, argname: str) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    "Return a user-friendly string representation of the given ``val``\n    that will be used by @pytest.mark.parametrize calls, or None if the hook\n    doesn't know about ``val``.\n\n    The parameter name is available as ``argname``, if required.\n\n    Stops at first non-None result, see :ref:`firstresult`.\n\n    :param config: The pytest config object.\n    :param val: The parametrized value.\n    :param str argname: The automatic parameter name produced by pytest.\n    "

@hookspec(firstresult=True)
def pytest_runtestloop(session: 'Session') -> Optional[object]:
    if False:
        i = 10
        return i + 15
    'Perform the main runtest loop (after collection finished).\n\n    The default hook implementation performs the runtest protocol for all items\n    collected in the session (``session.items``), unless the collection failed\n    or the ``collectonly`` pytest option is set.\n\n    If at any point :py:func:`pytest.exit` is called, the loop is\n    terminated immediately.\n\n    If at any point ``session.shouldfail`` or ``session.shouldstop`` are set, the\n    loop is terminated after the runtest protocol for the current item is finished.\n\n    :param session: The pytest session object.\n\n    Stops at first non-None result, see :ref:`firstresult`.\n    The return value is not used, but only stops further processing.\n    '

@hookspec(firstresult=True)
def pytest_runtest_protocol(item: 'Item', nextitem: 'Optional[Item]') -> Optional[object]:
    if False:
        print('Hello World!')
    'Perform the runtest protocol for a single test item.\n\n    The default runtest protocol is this (see individual hooks for full details):\n\n    - ``pytest_runtest_logstart(nodeid, location)``\n\n    - Setup phase:\n        - ``call = pytest_runtest_setup(item)`` (wrapped in ``CallInfo(when="setup")``)\n        - ``report = pytest_runtest_makereport(item, call)``\n        - ``pytest_runtest_logreport(report)``\n        - ``pytest_exception_interact(call, report)`` if an interactive exception occurred\n\n    - Call phase, if the the setup passed and the ``setuponly`` pytest option is not set:\n        - ``call = pytest_runtest_call(item)`` (wrapped in ``CallInfo(when="call")``)\n        - ``report = pytest_runtest_makereport(item, call)``\n        - ``pytest_runtest_logreport(report)``\n        - ``pytest_exception_interact(call, report)`` if an interactive exception occurred\n\n    - Teardown phase:\n        - ``call = pytest_runtest_teardown(item, nextitem)`` (wrapped in ``CallInfo(when="teardown")``)\n        - ``report = pytest_runtest_makereport(item, call)``\n        - ``pytest_runtest_logreport(report)``\n        - ``pytest_exception_interact(call, report)`` if an interactive exception occurred\n\n    - ``pytest_runtest_logfinish(nodeid, location)``\n\n    :param item: Test item for which the runtest protocol is performed.\n    :param nextitem: The scheduled-to-be-next test item (or None if this is the end my friend).\n\n    Stops at first non-None result, see :ref:`firstresult`.\n    The return value is not used, but only stops further processing.\n    '

def pytest_runtest_logstart(nodeid: str, location: Tuple[str, Optional[int], str]) -> None:
    if False:
        i = 10
        return i + 15
    'Called at the start of running the runtest protocol for a single item.\n\n    See :hook:`pytest_runtest_protocol` for a description of the runtest protocol.\n\n    :param nodeid: Full node ID of the item.\n    :param location: A tuple of ``(filename, lineno, testname)``\n        where ``filename`` is a file path relative to ``config.rootpath``\n        and ``lineno`` is 0-based.\n    '

def pytest_runtest_logfinish(nodeid: str, location: Tuple[str, Optional[int], str]) -> None:
    if False:
        while True:
            i = 10
    'Called at the end of running the runtest protocol for a single item.\n\n    See :hook:`pytest_runtest_protocol` for a description of the runtest protocol.\n\n    :param nodeid: Full node ID of the item.\n    :param location: A tuple of ``(filename, lineno, testname)``\n        where ``filename`` is a file path relative to ``config.rootpath``\n        and ``lineno`` is 0-based.\n    '

def pytest_runtest_setup(item: 'Item') -> None:
    if False:
        while True:
            i = 10
    "Called to perform the setup phase for a test item.\n\n    The default implementation runs ``setup()`` on ``item`` and all of its\n    parents (which haven't been setup yet). This includes obtaining the\n    values of fixtures required by the item (which haven't been obtained\n    yet).\n\n    :param item:\n        The item.\n    "

def pytest_runtest_call(item: 'Item') -> None:
    if False:
        i = 10
        return i + 15
    'Called to run the test for test item (the call phase).\n\n    The default implementation calls ``item.runtest()``.\n\n    :param item:\n        The item.\n    '

def pytest_runtest_teardown(item: 'Item', nextitem: Optional['Item']) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Called to perform the teardown phase for a test item.\n\n    The default implementation runs the finalizers and calls ``teardown()``\n    on ``item`` and all of its parents (which need to be torn down). This\n    includes running the teardown phase of fixtures required by the item (if\n    they go out of scope).\n\n    :param item:\n        The item.\n    :param nextitem:\n        The scheduled-to-be-next test item (None if no further test item is\n        scheduled). This argument is used to perform exact teardowns, i.e.\n        calling just enough finalizers so that nextitem only needs to call\n        setup functions.\n    '

@hookspec(firstresult=True)
def pytest_runtest_makereport(item: 'Item', call: 'CallInfo[None]') -> Optional['TestReport']:
    if False:
        for i in range(10):
            print('nop')
    'Called to create a :class:`~pytest.TestReport` for each of\n    the setup, call and teardown runtest phases of a test item.\n\n    See :hook:`pytest_runtest_protocol` for a description of the runtest protocol.\n\n    :param item: The item.\n    :param call: The :class:`~pytest.CallInfo` for the phase.\n\n    Stops at first non-None result, see :ref:`firstresult`.\n    '

def pytest_runtest_logreport(report: 'TestReport') -> None:
    if False:
        print('Hello World!')
    'Process the :class:`~pytest.TestReport` produced for each\n    of the setup, call and teardown runtest phases of an item.\n\n    See :hook:`pytest_runtest_protocol` for a description of the runtest protocol.\n    '

@hookspec(firstresult=True)
def pytest_report_to_serializable(config: 'Config', report: Union['CollectReport', 'TestReport']) -> Optional[Dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    'Serialize the given report object into a data structure suitable for\n    sending over the wire, e.g. converted to JSON.\n\n    :param config: The pytest config object.\n    :param report: The report.\n    '

@hookspec(firstresult=True)
def pytest_report_from_serializable(config: 'Config', data: Dict[str, Any]) -> Optional[Union['CollectReport', 'TestReport']]:
    if False:
        return 10
    'Restore a report object previously serialized with\n    :hook:`pytest_report_to_serializable`.\n\n    :param config: The pytest config object.\n    '

@hookspec(firstresult=True)
def pytest_fixture_setup(fixturedef: 'FixtureDef[Any]', request: 'SubRequest') -> Optional[object]:
    if False:
        print('Hello World!')
    'Perform fixture setup execution.\n\n    :param fixturdef:\n        The fixture definition object.\n    :param request:\n        The fixture request object.\n    :returns:\n        The return value of the call to the fixture function.\n\n    Stops at first non-None result, see :ref:`firstresult`.\n\n    .. note::\n        If the fixture function returns None, other implementations of\n        this hook function will continue to be called, according to the\n        behavior of the :ref:`firstresult` option.\n    '

def pytest_fixture_post_finalizer(fixturedef: 'FixtureDef[Any]', request: 'SubRequest') -> None:
    if False:
        print('Hello World!')
    'Called after fixture teardown, but before the cache is cleared, so\n    the fixture result ``fixturedef.cached_result`` is still available (not\n    ``None``).\n\n    :param fixturdef:\n        The fixture definition object.\n    :param request:\n        The fixture request object.\n    '

def pytest_sessionstart(session: 'Session') -> None:
    if False:
        return 10
    'Called after the ``Session`` object has been created and before performing collection\n    and entering the run test loop.\n\n    :param session: The pytest session object.\n    '

def pytest_sessionfinish(session: 'Session', exitstatus: Union[int, 'ExitCode']) -> None:
    if False:
        while True:
            i = 10
    'Called after whole test run finished, right before returning the exit status to the system.\n\n    :param session: The pytest session object.\n    :param exitstatus: The status which pytest will return to the system.\n    '

def pytest_unconfigure(config: 'Config') -> None:
    if False:
        return 10
    'Called before test process is exited.\n\n    :param config: The pytest config object.\n    '

def pytest_assertrepr_compare(config: 'Config', op: str, left: object, right: object) -> Optional[List[str]]:
    if False:
        for i in range(10):
            print('nop')
    'Return explanation for comparisons in failing assert expressions.\n\n    Return None for no custom explanation, otherwise return a list\n    of strings. The strings will be joined by newlines but any newlines\n    *in* a string will be escaped. Note that all but the first line will\n    be indented slightly, the intention is for the first line to be a summary.\n\n    :param config: The pytest config object.\n    :param op: The operator, e.g. `"=="`, `"!="`, `"not in"`.\n    :param left: The left operand.\n    :param right: The right operand.\n    '

def pytest_assertion_pass(item: 'Item', lineno: int, orig: str, expl: str) -> None:
    if False:
        print('Hello World!')
    'Called whenever an assertion passes.\n\n    .. versionadded:: 5.0\n\n    Use this hook to do some processing after a passing assertion.\n    The original assertion information is available in the `orig` string\n    and the pytest introspected assertion information is available in the\n    `expl` string.\n\n    This hook must be explicitly enabled by the ``enable_assertion_pass_hook``\n    ini-file option:\n\n    .. code-block:: ini\n\n        [pytest]\n        enable_assertion_pass_hook=true\n\n    You need to **clean the .pyc** files in your project directory and interpreter libraries\n    when enabling this option, as assertions will require to be re-written.\n\n    :param item: pytest item object of current test.\n    :param lineno: Line number of the assert statement.\n    :param orig: String with the original assertion.\n    :param expl: String with the assert explanation.\n    '

def pytest_report_header(config: 'Config', start_path: Path, startdir: 'LEGACY_PATH') -> Union[str, List[str]]:
    if False:
        print('Hello World!')
    'Return a string or list of strings to be displayed as header info for terminal reporting.\n\n    :param config: The pytest config object.\n    :param start_path: The starting dir.\n    :param startdir: The starting dir (deprecated).\n\n    .. note::\n\n        Lines returned by a plugin are displayed before those of plugins which\n        ran before it.\n        If you want to have your line(s) displayed first, use\n        :ref:`trylast=True <plugin-hookorder>`.\n\n    .. note::\n\n        This function should be implemented only in plugins or ``conftest.py``\n        files situated at the tests root directory due to how pytest\n        :ref:`discovers plugins during startup <pluginorder>`.\n\n    .. versionchanged:: 7.0.0\n        The ``start_path`` parameter was added as a :class:`pathlib.Path`\n        equivalent of the ``startdir`` parameter. The ``startdir`` parameter\n        has been deprecated.\n    '

def pytest_report_collectionfinish(config: 'Config', start_path: Path, startdir: 'LEGACY_PATH', items: Sequence['Item']) -> Union[str, List[str]]:
    if False:
        print('Hello World!')
    'Return a string or list of strings to be displayed after collection\n    has finished successfully.\n\n    These strings will be displayed after the standard "collected X items" message.\n\n    .. versionadded:: 3.2\n\n    :param config: The pytest config object.\n    :param start_path: The starting dir.\n    :param startdir: The starting dir (deprecated).\n    :param items: List of pytest items that are going to be executed; this list should not be modified.\n\n    .. note::\n\n        Lines returned by a plugin are displayed before those of plugins which\n        ran before it.\n        If you want to have your line(s) displayed first, use\n        :ref:`trylast=True <plugin-hookorder>`.\n\n    .. versionchanged:: 7.0.0\n        The ``start_path`` parameter was added as a :class:`pathlib.Path`\n        equivalent of the ``startdir`` parameter. The ``startdir`` parameter\n        has been deprecated.\n    '

@hookspec(firstresult=True)
def pytest_report_teststatus(report: Union['CollectReport', 'TestReport'], config: 'Config') -> 'TestShortLogReport | Tuple[str, str, Union[str, Tuple[str, Mapping[str, bool]]]]':
    if False:
        while True:
            i = 10
    'Return result-category, shortletter and verbose word for status\n    reporting.\n\n    The result-category is a category in which to count the result, for\n    example "passed", "skipped", "error" or the empty string.\n\n    The shortletter is shown as testing progresses, for example ".", "s",\n    "E" or the empty string.\n\n    The verbose word is shown as testing progresses in verbose mode, for\n    example "PASSED", "SKIPPED", "ERROR" or the empty string.\n\n    pytest may style these implicitly according to the report outcome.\n    To provide explicit styling, return a tuple for the verbose word,\n    for example ``"rerun", "R", ("RERUN", {"yellow": True})``.\n\n    :param report: The report object whose status is to be returned.\n    :param config: The pytest config object.\n    :returns: The test status.\n\n    Stops at first non-None result, see :ref:`firstresult`.\n    '

def pytest_terminal_summary(terminalreporter: 'TerminalReporter', exitstatus: 'ExitCode', config: 'Config') -> None:
    if False:
        print('Hello World!')
    'Add a section to terminal summary reporting.\n\n    :param terminalreporter: The internal terminal reporter object.\n    :param exitstatus: The exit status that will be reported back to the OS.\n    :param config: The pytest config object.\n\n    .. versionadded:: 4.2\n        The ``config`` parameter.\n    '

@hookspec(historic=True)
def pytest_warning_recorded(warning_message: 'warnings.WarningMessage', when: "Literal['config', 'collect', 'runtest']", nodeid: str, location: Optional[Tuple[str, int, str]]) -> None:
    if False:
        print('Hello World!')
    'Process a warning captured by the internal pytest warnings plugin.\n\n    :param warning_message:\n        The captured warning. This is the same object produced by :py:func:`warnings.catch_warnings`, and contains\n        the same attributes as the parameters of :py:func:`warnings.showwarning`.\n\n    :param when:\n        Indicates when the warning was captured. Possible values:\n\n        * ``"config"``: during pytest configuration/initialization stage.\n        * ``"collect"``: during test collection.\n        * ``"runtest"``: during test execution.\n\n    :param nodeid:\n        Full id of the item.\n\n    :param location:\n        When available, holds information about the execution context of the captured\n        warning (filename, linenumber, function). ``function`` evaluates to <module>\n        when the execution context is at the module level.\n\n    .. versionadded:: 6.0\n    '

def pytest_markeval_namespace(config: 'Config') -> Dict[str, Any]:
    if False:
        print('Hello World!')
    'Called when constructing the globals dictionary used for\n    evaluating string conditions in xfail/skipif markers.\n\n    This is useful when the condition for a marker requires\n    objects that are expensive or impossible to obtain during\n    collection time, which is required by normal boolean\n    conditions.\n\n    .. versionadded:: 6.2\n\n    :param config: The pytest config object.\n    :returns: A dictionary of additional globals to add.\n    '

def pytest_internalerror(excrepr: 'ExceptionRepr', excinfo: 'ExceptionInfo[BaseException]') -> Optional[bool]:
    if False:
        i = 10
        return i + 15
    'Called for internal errors.\n\n    Return True to suppress the fallback handling of printing an\n    INTERNALERROR message directly to sys.stderr.\n\n    :param excrepr: The exception repr object.\n    :param excinfo: The exception info.\n    '

def pytest_keyboard_interrupt(excinfo: 'ExceptionInfo[Union[KeyboardInterrupt, Exit]]') -> None:
    if False:
        i = 10
        return i + 15
    'Called for keyboard interrupt.\n\n    :param excinfo: The exception info.\n    '

def pytest_exception_interact(node: Union['Item', 'Collector'], call: 'CallInfo[Any]', report: Union['CollectReport', 'TestReport']) -> None:
    if False:
        i = 10
        return i + 15
    'Called when an exception was raised which can potentially be\n    interactively handled.\n\n    May be called during collection (see :hook:`pytest_make_collect_report`),\n    in which case ``report`` is a :class:`CollectReport`.\n\n    May be called during runtest of an item (see :hook:`pytest_runtest_protocol`),\n    in which case ``report`` is a :class:`TestReport`.\n\n    This hook is not called if the exception that was raised is an internal\n    exception like ``skip.Exception``.\n\n    :param node:\n        The item or collector.\n    :param call:\n        The call information. Contains the exception.\n    :param report:\n        The collection or test report.\n    '

def pytest_enter_pdb(config: 'Config', pdb: 'pdb.Pdb') -> None:
    if False:
        while True:
            i = 10
    'Called upon pdb.set_trace().\n\n    Can be used by plugins to take special action just before the python\n    debugger enters interactive mode.\n\n    :param config: The pytest config object.\n    :param pdb: The Pdb instance.\n    '

def pytest_leave_pdb(config: 'Config', pdb: 'pdb.Pdb') -> None:
    if False:
        i = 10
        return i + 15
    'Called when leaving pdb (e.g. with continue after pdb.set_trace()).\n\n    Can be used by plugins to take special action just after the python\n    debugger leaves interactive mode.\n\n    :param config: The pytest config object.\n    :param pdb: The Pdb instance.\n    '