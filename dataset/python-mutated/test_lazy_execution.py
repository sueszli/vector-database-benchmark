import jittor as jt
import unittest
import sys, os
from subprocess import getoutput

class TestLazyExecution(unittest.TestCase):

    @unittest.skipIf(not jt.has_cuda, 'No cuda found')
    def test_lazy_execution(self):
        if False:
            i = 10
            return i + 15
        code = "\nimport jittor as jt\njt.flags.use_cuda = 1\n\na = jt.zeros(1)\nb = jt.code([1], a.dtype, [a],\ncuda_header='''\n#include <assert.h>\n''',\ncuda_src='''\n__global__ void kernel(float32* a, float32* b) {\n    b[0] = a[0];\n    assert(a[0] == 1);\n}\nkernel<<<1,1>>>(in0_p, out0_p);\n''')\nc = a+b\nprint(c)\n"
        fpath = os.path.join(jt.flags.cache_path, 'lazy_error.py')
        with open(fpath, 'w') as f:
            f.write(code)
        res = getoutput(f'{sys.executable} {fpath}')
        assert 'print(c)' in res
        res = getoutput(f'lazy_execution=0 {sys.executable} {fpath}')
        assert "''')" in res
if __name__ == '__main__':
    unittest.main()