"""Tests for testlib_runner."""
import pathlib
from absl.testing import absltest
import numpy as np
from xla.runtime.runner import runner
r = runner.Runner(f'{pathlib.Path(__file__).parent.resolve()}/testlib_runner')

class TestlibRunnerTest(absltest.TestCase):

    def testScalarAdd(self):
        if False:
            for i in range(10):
                print('nop')
        module = '\n      func.func @add(%arg0: i32) -> i32 {\n        %0 = arith.constant 42 : i32\n        %1 = arith.addi %arg0, %0 : i32\n        return %1 : i32\n      }'
        [res] = r.execute(module, 'add', [42])
        self.assertEqual(res, 84)

    def testTensorAdd(self):
        if False:
            print('Hello World!')
        module = '\n      func.func @addtensor(%arg0: memref<?xf32>) {\n        %c0 = arith.constant 0 : index\n        %c1 = arith.constant 3 : index\n        %step = arith.constant 1 : index\n\n        scf.for %i = %c0 to %c1 step %step {\n          %0 = arith.constant 42.0 : f32\n          %1 = memref.load %arg0[%i] : memref<?xf32>\n          %2 = arith.addf %0, %1 : f32\n          memref.store %2, %arg0[%i] : memref<?xf32>\n        }\n        \n        func.return\n      }'
        arg = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        [res] = r.execute(module, 'addtensor', [arg], inout=[0])
        self.assertTrue(np.array_equal(res, np.array([43.0, 44.0, 45.0], dtype=np.float32)))

    def testTensorReturn(self):
        if False:
            while True:
                i = 10
        module = '\n      func.func @returntensor(%arg0: memref<?xf32>) -> memref<4xf32> {\n      %out = memref.alloc() : memref<4xf32>\n      %c0 = arith.constant 0 : index\n      %c1 = arith.constant 4 : index\n      %step = arith.constant 1 : index\n\n      scf.for %i = %c0 to %c1 step %step {\n        %0 = memref.load %arg0[%i] : memref<?xf32>\n        memref.store %0, %out[%i] : memref<4xf32>\n      }\n\n      return %out : memref<4xf32>\n    }'
        arg = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        [res] = r.execute(module, 'returntensor', [arg])
        self.assertTrue(np.array_equal(res, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)))
if __name__ == '__main__':
    absltest.main()