"""Run testsuites written for nose."""
import warnings
from _pytest.config import hookimpl
from _pytest.deprecated import NOSE_SUPPORT
from _pytest.fixtures import getfixturemarker
from _pytest.nodes import Item
from _pytest.python import Function
from _pytest.unittest import TestCaseFunction

@hookimpl(trylast=True)
def pytest_runtest_setup(item: Item) -> None:
    if False:
        while True:
            i = 10
    if not isinstance(item, Function):
        return
    if isinstance(item, TestCaseFunction):
        return
    func = item
    call_optional(func.obj, 'setup', func.nodeid)
    func.addfinalizer(lambda : call_optional(func.obj, 'teardown', func.nodeid))

def call_optional(obj: object, name: str, nodeid: str) -> bool:
    if False:
        i = 10
        return i + 15
    method = getattr(obj, name, None)
    if method is None:
        return False
    is_fixture = getfixturemarker(method) is not None
    if is_fixture:
        return False
    if not callable(method):
        return False
    method_name = getattr(method, '__name__', str(method))
    warnings.warn(NOSE_SUPPORT.format(nodeid=nodeid, method=method_name, stage=name), stacklevel=2)
    method()
    return True