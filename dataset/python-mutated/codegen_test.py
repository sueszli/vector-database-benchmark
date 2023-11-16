"""Tests for type_info module."""
import numpy as np
from tensorflow.python.autograph.pyct import compiler
from tensorflow.python.autograph.pyct.testing import codegen
from tensorflow.python.platform import test

class CodeGenTest(test.TestCase):

    def test_codegen_gens(self):
        if False:
            print('Hello World!')
        np.random.seed(0)
        for _ in range(1000):
            node = codegen.generate_random_functiondef()
            fn = compiler.ast_to_object(node)
            self.assertIsNotNone(fn, 'Generated invalid AST that could not convert to source.')
if __name__ == '__main__':
    test.main()