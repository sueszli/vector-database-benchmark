import re
import warnings
from pathlib import Path
from typing import cast
from typing import List
from typing import Type
import pytest
from _pytest import nodes
from _pytest.compat import legacy_path
from _pytest.outcomes import OutcomeException
from _pytest.pytester import Pytester
from _pytest.warning_types import PytestWarning

@pytest.mark.parametrize(('nodeid', 'expected'), (('', ['']), ('a', ['', 'a']), ('aa/b', ['', 'aa', 'aa/b']), ('a/b/c', ['', 'a', 'a/b', 'a/b/c']), ('a/bbb/c::D', ['', 'a', 'a/bbb', 'a/bbb/c', 'a/bbb/c::D']), ('a/b/c::D::eee', ['', 'a', 'a/b', 'a/b/c', 'a/b/c::D', 'a/b/c::D::eee']), ('::xx', ['', '::xx']), ('a/b/c::D/d::e', ['', 'a', 'a/b', 'a/b/c', 'a/b/c::D/d', 'a/b/c::D/d::e']), ('a/b::D:e:f::g', ['', 'a', 'a/b', 'a/b::D:e:f', 'a/b::D:e:f::g']), ('a/b::c/d::e[/test]', ['', 'a', 'a/b', 'a/b::c/d', 'a/b::c/d::e[/test]'])))
def test_iterparentnodeids(nodeid: str, expected: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    result = list(nodes.iterparentnodeids(nodeid))
    assert result == expected

def test_node_from_parent_disallowed_arguments() -> None:
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError, match='session is'):
        nodes.Node.from_parent(None, session=None)
    with pytest.raises(TypeError, match='config is'):
        nodes.Node.from_parent(None, config=None)

def test_node_direct_construction_deprecated() -> None:
    if False:
        while True:
            i = 10
    with pytest.raises(OutcomeException, match='Direct construction of _pytest.nodes.Node has been deprecated, please use _pytest.nodes.Node.from_parent.\nSee https://docs.pytest.org/en/stable/deprecations.html#node-construction-changed-to-node-from-parent for more details.'):
        nodes.Node(None, session=None)

def test_subclassing_both_item_and_collector_deprecated(request, tmp_path: Path) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Verifies we warn on diamond inheritance as well as correctly managing legacy\n    inheritance constructors with missing args as found in plugins.\n    '
    with warnings.catch_warnings():
        warnings.simplefilter('error')

        class SoWrong(nodes.Item, nodes.File):

            def __init__(self, fspath, parent):
                if False:
                    return 10
                "Legacy ctor with legacy call # don't wana see"
                super().__init__(fspath, parent)
    with pytest.warns(PytestWarning) as rec:
        SoWrong.from_parent(request.session, fspath=legacy_path(tmp_path / 'broken.txt'))
    messages = [str(x.message) for x in rec]
    assert any((re.search('.*SoWrong.* not using a cooperative constructor.*', x) for x in messages))
    assert any((re.search('(?m)SoWrong .* should not be a collector', x) for x in messages))

@pytest.mark.parametrize('warn_type, msg', [(DeprecationWarning, 'deprecated'), (PytestWarning, 'pytest')])
def test_node_warn_is_no_longer_only_pytest_warnings(pytester: Pytester, warn_type: Type[Warning], msg: str) -> None:
    if False:
        return 10
    items = pytester.getitems('\n        def test():\n            pass\n    ')
    with pytest.warns(warn_type, match=msg):
        items[0].warn(warn_type(msg))

def test_node_warning_enforces_warning_types(pytester: Pytester) -> None:
    if False:
        return 10
    items = pytester.getitems('\n        def test():\n            pass\n    ')
    with pytest.raises(ValueError, match='warning must be an instance of Warning or subclass'):
        items[0].warn(Exception('ok'))

def test__check_initialpaths_for_relpath() -> None:
    if False:
        i = 10
        return i + 15
    'Ensure that it handles dirs, and does not always use dirname.'
    cwd = Path.cwd()

    class FakeSession1:
        _initialpaths = frozenset({cwd})
    session = cast(pytest.Session, FakeSession1)
    assert nodes._check_initialpaths_for_relpath(session, cwd) == ''
    sub = cwd / 'file'

    class FakeSession2:
        _initialpaths = frozenset({cwd})
    session = cast(pytest.Session, FakeSession2)
    assert nodes._check_initialpaths_for_relpath(session, sub) == 'file'
    outside = Path('/outside-this-does-not-exist')
    assert nodes._check_initialpaths_for_relpath(session, outside) is None

def test_failure_with_changed_cwd(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test failure lines should use absolute paths if cwd has changed since\n    invocation, so the path is correct (#6428).\n    '
    p = pytester.makepyfile("\n        import os\n        import pytest\n\n        @pytest.fixture\n        def private_dir():\n            out_dir = 'ddd'\n            os.mkdir(out_dir)\n            old_dir = os.getcwd()\n            os.chdir(out_dir)\n            yield out_dir\n            os.chdir(old_dir)\n\n        def test_show_wrong_path(private_dir):\n            assert False\n    ")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines([str(p) + ':*: AssertionError', '*1 failed in *'])