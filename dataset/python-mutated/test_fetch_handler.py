import time
import unittest
import numpy as np
from paddle import base
from paddle.base import core
from paddle.base.framework import Program

class TestFetchHandler(unittest.TestCase):

    @unittest.skip(reason='Skip unstable ci')
    def test_fetch_handler(self):
        if False:
            return 10
        place = core.CPUPlace()
        scope = core.Scope()
        table = np.random.random((3, 10)).astype('float32')
        prog = Program()
        block = prog.current_block()
        var_emb = block.create_var(name='emb', type=core.VarDesc.VarType.FP32)
        var_emb3 = block.create_var(name='emb3', type=core.VarDesc.VarType.FP32)

        class FH(base.executor.FetchHandler):

            def handler(self, fetch_dict):
                if False:
                    for i in range(10):
                        print('nop')
                assert len(fetch_dict) == 1
        table_var = scope.var('emb').get_tensor()
        table_var.set(table, place)
        fh = FH(var_dict={'emb': var_emb}, period_secs=2)
        fm = base.trainer_factory.FetchHandlerMonitor(scope, fh)
        fm.start()
        time.sleep(3)
        fm.stop()
        default_fh = base.executor.FetchHandler(var_dict={'emb': var_emb, 'emb2': None, 'emb3': var_emb3}, period_secs=1)
        default_fm = base.trainer_factory.FetchHandlerMonitor(scope, default_fh)
        default_fm.start()
        time.sleep(5)
        default_fm.stop()
if __name__ == '__main__':
    unittest.main()