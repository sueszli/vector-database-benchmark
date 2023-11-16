import logging
import os
import pathlib
import re
import shutil
import sys
import unittest
import torch
from torch._inductor import config, test_operators
try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor
except unittest.SkipTest:
    if __name__ == '__main__':
        sys.exit(0)
    raise

def filesize(filename: pathlib.Path):
    if False:
        i = 10
        return i + 15
    assert filename.exists(), f'{filename} is missing'
    return os.stat(filename).st_size

@config.patch('trace.enabled', True)
class TestDebugTrace(test_torchinductor.TestCase):

    def test_debug_trace(self):
        if False:
            i = 10
            return i + 15

        @torch.compile
        def fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            a = test_operators.realize(a + 1) + 2
            return torch.matmul(a, b)
        with self.assertLogs(logging.getLogger('torch._inductor.debug'), level=logging.WARNING) as cm:
            fn(torch.randn(16, 16), torch.randn(16, 16))
        self.assertEqual(len(cm.output), 1)
        m = re.match('WARNING.* debug trace: (.*)', cm.output[0])
        self.assertTrue(m)
        filename = pathlib.Path(m.group(1))
        self.assertTrue(filename.is_dir())
        self.assertGreater(filesize(filename / 'fx_graph_readable.py'), 512)
        self.assertGreater(filesize(filename / 'fx_graph_runnable.py'), 512)
        self.assertGreater(filesize(filename / 'fx_graph_transformed.py'), 512)
        self.assertGreater(filesize(filename / 'output_code.py'), 1024)
        self.assertExpectedInline(open(filename / 'ir_pre_fusion.txt').read().rstrip(), "buf0: SchedulerNode(ComputedBuffer)\nbuf0.writes = [MemoryDep('buf0', c0, {c0: 256})]\nbuf0.unmet_dependencies = []\nbuf0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256})]\nbuf0.users = [NodeUser(node=SchedulerNode(name='buf1'), can_inplace=True, is_weak=False)]\nbuf0.group.device = cpu\nbuf0.group.iteration = ((256,), ())\nbuf0.sizes = ([256], [])\nclass buf0_loop_body:\n    var_ranges = {z0: 256}\n    index0 = z0\n    def body(self, ops):\n        get_index = self.get_index('index0')\n        load = ops.load('arg0_1', get_index)\n        constant = ops.constant(1.0, torch.float32)\n        add = ops.add(load, constant)\n        get_index_1 = self.get_index('index0')\n        store = ops.store('buf0', get_index_1, add, None)\n        return store\n\n\nbuf1: SchedulerNode(ComputedBuffer)\nbuf1.writes = [MemoryDep('buf1', c0, {c0: 256})]\nbuf1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 256})]\nbuf1.met_dependencies = []\nbuf1.users = [NodeUser(node=ExternKernelSchedulerNode(name='buf2'), can_inplace=False, is_weak=False)]\nbuf1.group.device = cpu\nbuf1.group.iteration = ((256,), ())\nbuf1.sizes = ([256], [])\nclass buf1_loop_body:\n    var_ranges = {z0: 256}\n    index0 = z0\n    def body(self, ops):\n        get_index = self.get_index('index0')\n        load = ops.load('buf0', get_index)\n        constant = ops.constant(2.0, torch.float32)\n        add = ops.add(load, constant)\n        get_index_1 = self.get_index('index0')\n        store = ops.store('buf1', get_index_1, add, None)\n        return store\n\n\nbuf2: ExternKernelSchedulerNode(ExternKernelOut)\nbuf2.writes = [StarDep(name='buf2')]\nbuf2.unmet_dependencies = [StarDep(name='buf1')]\nbuf2.met_dependencies = [StarDep(name='arg1_1')]\nbuf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]\nbuf2.node.kernel = extern_kernels.mm")
        self.assertExpectedInline(open(filename / 'ir_post_fusion.txt').read().rstrip(), "buf0_buf1: FusedSchedulerNode(NoneType)\nbuf0_buf1.writes = [MemoryDep('buf0', c0, {c0: 256}), MemoryDep('buf1', c0, {c0: 256})]\nbuf0_buf1.unmet_dependencies = []\nbuf0_buf1.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256})]\nbuf0_buf1.users = []\n    buf0_buf1.snodes[0] =\n    buf0: SchedulerNode(ComputedBuffer)\n    buf0.writes = [MemoryDep('buf0', c0, {c0: 256})]\n    buf0.unmet_dependencies = []\n    buf0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256})]\n    buf0.users = [NodeUser(node=SchedulerNode(name='buf1'), can_inplace=True, is_weak=False)]\n    buf0.group.device = cpu\n    buf0.group.iteration = ((256,), ())\n    buf0.sizes = ([256], [])\n    class buf0_loop_body:\n        var_ranges = {z0: 256}\n        index0 = z0\n        def body(self, ops):\n            get_index = self.get_index('index0')\n            load = ops.load('arg0_1', get_index)\n            constant = ops.constant(1.0, torch.float32)\n            add = ops.add(load, constant)\n            get_index_1 = self.get_index('index0')\n            store = ops.store('buf0', get_index_1, add, None)\n            return store\n    buf0_buf1.snodes[1] =\n    buf1: SchedulerNode(ComputedBuffer)\n    buf1.writes = [MemoryDep('buf1', c0, {c0: 256})]\n    buf1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 256})]\n    buf1.met_dependencies = []\n    buf1.users = [NodeUser(node=ExternKernelSchedulerNode(name='buf2'), can_inplace=False, is_weak=False)]\n    buf1.group.device = cpu\n    buf1.group.iteration = ((256,), ())\n    buf1.sizes = ([256], [])\n    class buf1_loop_body:\n        var_ranges = {z0: 256}\n        index0 = z0\n        def body(self, ops):\n            get_index = self.get_index('index0')\n            load = ops.load('buf0', get_index)\n            constant = ops.constant(2.0, torch.float32)\n            add = ops.add(load, constant)\n            get_index_1 = self.get_index('index0')\n            store = ops.store('buf1', get_index_1, add, None)\n            return store\n\n\nbuf2: ExternKernelSchedulerNode(ExternKernelOut)\nbuf2.writes = [StarDep(name='buf2')]\nbuf2.unmet_dependencies = [StarDep(name='buf1')]\nbuf2.met_dependencies = [StarDep(name='arg1_1')]\nbuf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]\nbuf2.node.kernel = extern_kernels.mm")
        shutil.rmtree(filename)
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU
    if HAS_CPU:
        run_tests(needs='filelock')