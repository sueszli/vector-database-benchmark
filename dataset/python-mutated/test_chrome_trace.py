import os
import json
import unittest
from textwrap import dedent
from tempfile import TemporaryDirectory
from numba.tests.support import TestCase, run_in_subprocess

class TestChromeTraceModule(TestCase):
    """
    Test chrome tracing generated file(s).
    """

    def test_trace_output(self):
        if False:
            print('Hello World!')
        code = '\n            from numba import njit\n            import numpy as np\n\n            x = np.arange(100).reshape(10, 10)\n\n            @njit\n            def go_fast(a):\n                trace = 0.0\n                for i in range(a.shape[0]):\n                    trace += np.tanh(a[i, i])\n                return a + trace\n\n            go_fast(x)\n        '
        src = dedent(code)
        with TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_trace.json')
            env = os.environ.copy()
            env['NUMBA_CHROME_TRACE'] = path
            run_in_subprocess(src, env=env)
            with open(path) as file:
                events = json.load(file)
                self.assertIsInstance(events, list)
                for ev in events:
                    self.assertIsInstance(ev, dict)
                    self.assertEqual(set(ev.keys()), {'cat', 'pid', 'tid', 'ph', 'name', 'args', 'ts'})
if __name__ == '__main__':
    unittest.main()