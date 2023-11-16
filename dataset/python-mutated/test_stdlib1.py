"""Tests of selected stdlib functions."""
from pytype.tests import test_base
from pytype.tests import test_utils

class StdlibTests(test_base.BaseTest):
    """Tests for files in typeshed/stdlib."""

    def test_ast(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import ast\n      def f():\n        return ast.parse("True")\n    ')
        self.assertTypesMatchPytd(ty, '\n      import ast\n      def f() -> _ast.Module: ...\n    ')

    def test_urllib(self):
        if False:
            return 10
        ty = self.Infer('\n      import urllib\n    ')
        self.assertTypesMatchPytd(ty, '\n      import urllib\n    ')

    def test_traceback(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import traceback\n      def f(exc):\n        return traceback.format_exception(*exc)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import traceback\n      from typing import List\n      def f(exc) -> List[str]: ...\n    ')

    def test_os_walk(self):
        if False:
            return 10
        ty = self.Infer('\n      import os\n      x = list(os.walk("/tmp"))\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      import os\n      from typing import List, Tuple\n      x = ...  # type: List[Tuple[str, List[str], List[str]]]\n    ')

    def test_struct(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import struct\n      x = struct.Struct("b")\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      import struct\n      x = ...  # type: struct.Struct\n    ')

    def test_warning(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import warnings\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      import warnings\n    ')

    @test_utils.skipOnWin32('os.pathconf is not supported on Windows')
    def test_path_conf(self):
        if False:
            print('Hello World!')
        self.Check("\n      import os\n      max_len = os.pathconf('directory', 'name')\n      filename = 'foobar.baz'\n      r = len(filename) >= max_len - 1\n    ")

    def test_environ(self):
        if False:
            print('Hello World!')
        self.Check("\n      import os\n      os.getenv('foobar', 3j)\n      os.environ['hello'] = 'bar'\n      x = os.environ['hello']\n      y = os.environ.get(3.14, None)\n      z = os.environ.get(3.14, 3j)\n      del os.environ['hello']\n    ")

    def test_stdlib(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import re\n      s = "the quick brown fox jumps over the lazy dog"\n      word = re.compile(r"\\w*")\n      word.sub(lambda x: \'<\'+x.group(0)+\'>\', s)\n    ')

    def test_namedtuple(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import collections\n      collections.namedtuple(u"_", "")\n      collections.namedtuple("_", u"")\n      collections.namedtuple("_", [u"a", "b"])\n    ')

    def test_defaultdict(self):
        if False:
            return 10
        ty = self.Infer("\n      import collections\n      a = collections.defaultdict(int, one = 1, two = 2)\n      b = collections.defaultdict(int, {'one': 1, 'two': 2})\n      c = collections.defaultdict(int, [('one', 1), ('two', 2)])\n      d = collections.defaultdict(int, {})\n      e = collections.defaultdict(int)\n      f = collections.defaultdict(default_factory = int)\n      ")
        self.assertTypesMatchPytd(ty, '\n      import collections\n      a = ...  # type: collections.defaultdict[str, int]\n      b = ...  # type: collections.defaultdict[str, int]\n      c = ...  # type: collections.defaultdict[str, int]\n      d = ...  # type: collections.defaultdict[nothing, int]\n      e = ...  # type: collections.defaultdict[nothing, int]\n      f = ...  # type: collections.defaultdict[nothing, int]\n      ')

    def test_defaultdict_no_factory(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      import collections\n      a = collections.defaultdict()\n      b = collections.defaultdict(None)\n      c = collections.defaultdict(lambda: __any_object__)\n      d = collections.defaultdict(None, one = 1, two = 2)\n      e = collections.defaultdict(None, {'one': 1, 'two': 2})\n      f = collections.defaultdict(None, [('one', 1), ('two', 2)])\n      g = collections.defaultdict(one = 1, two = 2)\n      h = collections.defaultdict(default_factory = None)\n      ")
        self.assertTypesMatchPytd(ty, '\n      import collections\n      from typing import Any\n      a = ...  # type: collections.defaultdict[nothing, nothing]\n      b = ...  # type: collections.defaultdict[nothing, nothing]\n      c = ...  # type: collections.defaultdict[nothing, Any]\n      d = ...  # type: collections.defaultdict[str, int]\n      e = ...  # type: collections.defaultdict[str, int]\n      f = ...  # type: collections.defaultdict[str, int]\n      g = ...  # type: collections.defaultdict[str, int]\n      h = ...  # type: collections.defaultdict[nothing, nothing]\n      ')

    def test_defaultdict_diff_defaults(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      import collections\n      a = collections.defaultdict(int, one = '1')\n      b = collections.defaultdict(str, one = 1)\n      c = collections.defaultdict(None, one = 1)\n      d = collections.defaultdict(int, {1: 'one'})\n      ")
        self.assertTypesMatchPytd(ty, '\n      import collections\n      from typing import Union\n      a = ...  # type: collections.defaultdict[str, Union[int, str]]\n      b = ...  # type: collections.defaultdict[str, Union[int, str]]\n      c = ...  # type: collections.defaultdict[str, int]\n      d = ...  # type: collections.defaultdict[int, Union[int, str]]\n      ')

    def test_counter(self):
        if False:
            print('Hello World!')
        self.Check('\n      import collections\n      x = collections.Counter()\n      y = collections.Counter()\n      (x + y).elements\n      (x - y).elements\n      (x & y).elements\n      (x | y).elements\n    ')

    def test_range(self):
        if False:
            print('Hello World!')
        self.Check('\n      import random\n      random.sample(range(10), 5)\n    ')

    def test_xml(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import xml.etree.cElementTree\n      xml.etree.cElementTree.SubElement\n      xml.etree.cElementTree.iterparse\n    ')

    def test_csv(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import _csv\n      import csv\n    ')

    def test_future(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import __future__\n    ')

    def test_sys_version_info(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import sys\n      major, minor, micro, releaselevel, serial = sys.version_info\n    ')
        self.assertTypesMatchPytd(ty, '\n      import sys\n      major: int\n      minor: int\n      micro: int\n      releaselevel: str\n      serial: int\n    ')

    def test_subprocess(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import subprocess\n      def run(cmd):\n        proc = subprocess.Popen(cmd)\n        return proc.communicate()\n    ')

    def test_subprocess_subclass(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import subprocess\n      class Popen(subprocess.Popen):\n        def wait(self, *args, **kwargs):\n          return super(Popen, self).wait(*args, **kwargs)\n    ')

    def test_subprocess_src_and_pyi(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import subprocess\n        def f() -> subprocess.Popen: ...\n      ')
            self.Check('\n        import foo\n        import subprocess\n\n        def f():\n          p = foo.f()\n          return p.communicate()\n\n        def g():\n          p = subprocess.Popen(__any_object__)\n          return p.communicate()\n      ', pythonpath=[d.path])

    def test_namedtuple_from_counter(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      import collections\n      import six\n      Foo = collections.namedtuple('Foo', ('x', 'y'))\n      def foo(self):\n        c = collections.Counter()\n        return [Foo(*x) for x in six.iteritems(c)]\n    ")

    def test_path(self):
        if False:
            while True:
                i = 10
        self.Check("\n      import pkgutil\n      __path__ = pkgutil.extend_path(__path__, '')\n    ")
if __name__ == '__main__':
    test_base.main()