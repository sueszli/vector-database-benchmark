import pytest
from ..helpers import test_dir

def test_import_references(Script):
    if False:
        i = 10
        return i + 15
    s = Script('from .. import foo', path=test_dir.joinpath('foo.py'))
    assert [usage.line for usage in s.get_references()] == [1]

def test_exclude_builtin_modules(Script):
    if False:
        while True:
            i = 10

    def get(include):
        if False:
            return 10
        references = Script(source).get_references(include_builtins=include)
        return [(d.line, d.column) for d in references]
    source = 'import sys\nsys.setprofile'
    places = get(include=True)
    assert len(places) >= 3
    places = get(include=False)
    assert places == [(2, 4)]

@pytest.mark.parametrize('code, places', [('', [(1, 7), (4, 6)]), ('', [(2, 5)]), ('', [(2, 24), (7, 10), (11, 10)]), ('', [(6, 4), (14, 0)]), ('', [(7, 4), (8, 11)]), ('', [(7, 22), (11, 22)]), ('', [(11, 4), (12, 11)]), ('from datetime', [(1, 5)]), ('from datetime import datetime\nd1 = datetime.now()\nd2 = datetime.now()\n', [(2, 14), (3, 14)]), ('from datetime import timedelta\nt1 = timedelta(seconds=1)\nt2 = timedelta(seconds=2)\n', [(2, 15), (3, 15)])])
def test_references_scope(Script, code, places):
    if False:
        print('Hello World!')
    if not code:
        code = 'import sys\nfrom collections import defaultdict\n\nprint(sys.path)\n\ndef foo(bar):\n    baz = defaultdict(int)\n    return baz\n\ndef bar(foo):\n    baz = defaultdict(int)\n    return baz\n\nfoo()\n'
    from jedi.api.project import Project
    project = Project('', sys_path=[], smart_sys_path=False)
    script = Script(code, project=project)
    for place in places:
        assert places == [(n.line, n.column) for n in script.get_references(*place, scope='file')]