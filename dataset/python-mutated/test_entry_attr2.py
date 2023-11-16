import paddle
paddle.enable_static()
import unittest
from paddle import base

class EntryAttrChecks(unittest.TestCase):

    def embedding_layer(self):
        if False:
            print('Hello World!')
        prog = base.Program()
        scope = base.core.Scope()
        with base.scope_guard(scope):
            with base.program_guard(prog):
                input = paddle.static.data(name='dnn_data', shape=[-1, 1], dtype='int64', lod_level=1)
                emb = paddle.static.nn.embedding(input=input, size=[100, 10], is_sparse=True, is_distributed=True, param_attr=base.ParamAttr(name='deep_embedding'))
                pool = paddle.static.nn.sequence_lod.sequence_pool(input=emb, pool_type='sum')
                predict = paddle.static.nn.fc(x=pool, size=2, activation='softmax')
        block = prog.global_block()
        for op in block.ops:
            if op.type == 'lookup_table':
                is_sparse = op.attr('is_sparse')
                is_distributed = op.attr('is_distributed')
                self.assertFalse(is_distributed)
                self.assertTrue(is_sparse)

class TestEntryAttrs(EntryAttrChecks):

    def test_embedding_layer(self):
        if False:
            for i in range(10):
                print('nop')
        self.embedding_layer()
if __name__ == '__main__':
    unittest.main()