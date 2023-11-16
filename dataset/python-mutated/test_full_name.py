"""
Tests for :attr:`.BaseName.full_name`.

There are three kinds of test:

#. Test classes derived from :class:`MixinTestFullName`.
   Child class defines :attr:`.operation` to alter how
   the api definition instance is created.

#. :class:`TestFullDefinedName` is to test combination of
   ``obj.full_name`` and ``jedi.defined_names``.

#. Misc single-function tests.
"""
import textwrap
from unittest import TestCase
import pytest
import jedi

class MixinTestFullName(object):
    operation = None

    @pytest.fixture(autouse=True)
    def init(self, Script, environment):
        if False:
            print('Hello World!')
        self.Script = Script
        self.environment = environment

    def check(self, source, desired):
        if False:
            return 10
        script = self.Script(textwrap.dedent(source))
        definitions = getattr(script, self.operation)()
        for d in definitions:
            self.assertEqual(d.full_name, desired)

    def test_os_path_join(self):
        if False:
            return 10
        self.check('import os; os.path.join', 'os.path.join')

    def test_builtin(self):
        if False:
            i = 10
            return i + 15
        self.check('TypeError', 'builtins.TypeError')

class TestFullNameWithGotoDefinitions(MixinTestFullName, TestCase):
    operation = 'infer'

    def test_tuple_mapping(self):
        if False:
            return 10
        self.check("\n        import re\n        any_re = re.compile('.*')\n        any_re", 'typing.Pattern')

    def test_from_import(self):
        if False:
            i = 10
            return i + 15
        self.check('from os import path', 'os.path')

class TestFullNameWithCompletions(MixinTestFullName, TestCase):
    operation = 'complete'

class TestFullDefinedName(TestCase):
    """
    Test combination of ``obj.full_name`` and ``jedi.Script.get_names``.
    """

    @pytest.fixture(autouse=True)
    def init(self, environment):
        if False:
            while True:
                i = 10
        self.environment = environment

    def check(self, source, desired):
        if False:
            for i in range(10):
                print('nop')
        script = jedi.Script(textwrap.dedent(source), environment=self.environment)
        definitions = script.get_names()
        full_names = [d.full_name for d in definitions]
        self.assertEqual(full_names, desired)

    def test_local_names(self):
        if False:
            for i in range(10):
                print('nop')
        self.check('\n        def f(): pass\n        class C: pass\n        ', ['__main__.f', '__main__.C'])

    def test_imports(self):
        if False:
            i = 10
            return i + 15
        self.check('\n        import os\n        from os import path\n        from os.path import join\n        from os import path as opath\n        ', ['os', 'os.path', 'os.path.join', 'os.path'])

def test_sub_module(Script, jedi_path):
    if False:
        return 10
    "\n    ``full_name needs to check sys.path to actually find it's real path module\n    path.\n    "
    sys_path = [jedi_path]
    project = jedi.Project('.', sys_path=sys_path)
    defs = Script('from jedi.api import classes; classes', project=project).infer()
    assert [d.full_name for d in defs] == ['jedi.api.classes']
    defs = Script('import jedi.api; jedi.api', project=project).infer()
    assert [d.full_name for d in defs] == ['jedi.api']

def test_os_path(Script):
    if False:
        i = 10
        return i + 15
    (d,) = Script('from os.path import join').complete()
    assert d.full_name == 'os.path.join'
    (d,) = Script('import os.p').complete()
    assert d.full_name == 'os.path'

def test_os_issues(Script):
    if False:
        for i in range(10):
            print('nop')
    'Issue #873'
    assert [c.name for c in Script('import os\nos.nt').complete()] == []

def test_param_name(Script):
    if False:
        i = 10
        return i + 15
    (name,) = Script('class X:\n def foo(bar): bar').goto()
    assert name.type == 'param'
    assert name.full_name is None

def test_variable_in_func(Script):
    if False:
        return 10
    names = Script('def f(): x = 3').get_names(all_scopes=True)
    x = names[-1]
    assert x.name == 'x'
    assert x.full_name == '__main__.f.x'