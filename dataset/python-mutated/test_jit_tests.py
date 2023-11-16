import unittest
import jittor as jt
from jittor import LOG

def test(name):
    if False:
        print('Hello World!')
    doc = eval(f'jt.tests.{name}.__doc__')
    doc = doc[doc.find('From'):].strip()
    LOG.i(f'Run test {name} {doc}')
    exec(f'jt.tests.{name}()')
tests = [name for name in dir(jt.tests) if not name.startswith('__')]
src = 'class TestJitTests(unittest.TestCase):\n'
for name in tests:
    doc = eval(f'jt.tests.{name}.__doc__')
    doc = doc[doc.find('From'):].strip()
    src += f'\n    def test_{name}(self):\n        test("{name}")\n    '
LOG.vvv('eval src\n' + src)
exec(src)
if __name__ == '__main__':
    unittest.main()