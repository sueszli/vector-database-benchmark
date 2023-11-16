"""Module containing a parametrized tests testing cross-python serialization
via the pickle module."""
import shutil
import subprocess
import textwrap
import pytest
pythonlist = ['python3.9', 'python3.10', 'python3.11']

@pytest.fixture(params=pythonlist)
def python1(request, tmp_path):
    if False:
        while True:
            i = 10
    picklefile = tmp_path / 'data.pickle'
    return Python(request.param, picklefile)

@pytest.fixture(params=pythonlist)
def python2(request, python1):
    if False:
        while True:
            i = 10
    return Python(request.param, python1.picklefile)

class Python:

    def __init__(self, version, picklefile):
        if False:
            print('Hello World!')
        self.pythonpath = shutil.which(version)
        if not self.pythonpath:
            pytest.skip(f'{version!r} not found')
        self.picklefile = picklefile

    def dumps(self, obj):
        if False:
            print('Hello World!')
        dumpfile = self.picklefile.with_name('dump.py')
        dumpfile.write_text(textwrap.dedent("\n                import pickle\n                f = open({!r}, 'wb')\n                s = pickle.dump({!r}, f, protocol=2)\n                f.close()\n                ".format(str(self.picklefile), obj)))
        subprocess.run((self.pythonpath, str(dumpfile)), check=True)

    def load_and_is_true(self, expression):
        if False:
            while True:
                i = 10
        loadfile = self.picklefile.with_name('load.py')
        loadfile.write_text(textwrap.dedent("\n                import pickle\n                f = open({!r}, 'rb')\n                obj = pickle.load(f)\n                f.close()\n                res = eval({!r})\n                if not res:\n                    raise SystemExit(1)\n                ".format(str(self.picklefile), expression)))
        print(loadfile)
        subprocess.run((self.pythonpath, str(loadfile)), check=True)

@pytest.mark.parametrize('obj', [42, {}, {1: 3}])
def test_basic_objects(python1, python2, obj):
    if False:
        for i in range(10):
            print('nop')
    python1.dumps(obj)
    python2.load_and_is_true(f'obj == {obj}')