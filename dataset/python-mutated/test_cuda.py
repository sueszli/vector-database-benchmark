import os
import sys
import gc
import unittest
import torch
from typing import NamedTuple
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_utils import skipIfRocm, skipCUDANonDefaultStreamIf, NoTest, TEST_CUDA
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2
if not TEST_CUDA:
    print('CUDA not available, skipping tests', file=sys.stderr)
    JitTestCase = NoTest
TEST_LARGE_TENSOR = TEST_CUDA
if TEST_CUDA:
    torch.ones(1).cuda()
    TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 5000000000.0
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestCUDA(JitTestCase):
    """
    A suite of tests for the CUDA API in TorchScript.
    """

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        gc.collect()
        torch.cuda.empty_cache()
        super().tearDown()

    @skipIfRocm
    @unittest.skipIf(not TEST_MULTIGPU, 'detected only one GPU')
    def test_cuda_synchronize(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def test_device_synchronize():
            if False:
                return 10
            prev_current_device_index = torch.cuda.current_device()
            torch.cuda.synchronize()
            torch.cuda.synchronize('cuda')
            torch.cuda.synchronize('cuda:0')
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(torch.device('cuda:1'))
            after_current_device_index = torch.cuda.current_device()
            return prev_current_device_index == after_current_device_index

        @torch.jit.script
        def test_multi_device_synchronize():
            if False:
                i = 10
                return i + 15
            torch.cuda.synchronize(torch.device('cuda:0'))
            prev_current_device_index = torch.cuda.current_device()
            torch.cuda.synchronize(1)
            after_current_device_index = torch.cuda.current_device()
            return prev_current_device_index == after_current_device_index
        self.assertTrue(test_device_synchronize)
        FileCheck().check('cuda::synchronize(').run(test_device_synchronize.graph)
        self.assertTrue(test_multi_device_synchronize)
        FileCheck().check('cuda::synchronize(').run(test_multi_device_synchronize.graph)

    def test_stream_args(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def stream_default_args() -> bool:
            if False:
                for i in range(10):
                    print('nop')
            s = torch.cuda.Stream()
            return s.device_index() == torch.cuda.current_device()

        @torch.jit.script
        def stream_default_args_for_device() -> bool:
            if False:
                for i in range(10):
                    print('nop')
            s = torch.cuda.Stream(priority=0)
            return s.device_index() == torch.cuda.current_device()

        @torch.jit.script
        def stream_default_args_for_priority() -> bool:
            if False:
                while True:
                    i = 10
            d = torch.device('cuda:1')
            s = torch.cuda.Stream(d)
            return s.device_index() == 1

        @torch.jit.script
        def stream_args_all() -> bool:
            if False:
                print('Hello World!')
            d = torch.device('cuda:0')
            s = torch.cuda.Stream(d, 0)
            return s.device_index() == 0
        self.assertTrue(stream_default_args)
        self.assertTrue(stream_default_args_for_device)
        self.assertTrue(stream_default_args_for_priority)
        self.assertTrue(stream_args_all)

    def test_event_args(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def event_default_args() -> bool:
            if False:
                return 10
            e = torch.cuda.Event()
            return e is not None
        self.assertTrue(event_default_args)

    @skipIfRocm
    @unittest.skipIf(not TEST_MULTIGPU, 'detected only one GPU')
    def test_current_stream(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn():
            if False:
                return 10
            device_index = torch.cuda.current_device()
            device = torch.device('cuda:' + str(device_index))
            s0 = torch.cuda.current_stream(device)
            s1 = torch.cuda.current_stream(torch.device('cuda:1'))
            s2 = torch.cuda.current_stream(torch.device('cuda:0'))
            return (s0.device_index(), s1.device_index(), s2.device_index())
        (d0, d1, d2) = fn()
        self.assertEqual(0, d0)
        self.assertEqual(1, d1)
        self.assertEqual(0, d2)
        self.assertEqual(d0, d2)

        @torch.jit.script
        def fn_with_device_index_args():
            if False:
                return 10
            device_index = torch.cuda.current_device()
            s0 = torch.cuda.current_stream(device_index)
            s1 = torch.cuda.current_stream(1)
            s2 = torch.cuda.current_stream(0)
            return (s0.device_index(), s1.device_index(), s2.device_index())
        (d0, d1, d2) = fn_with_device_index_args()
        self.assertEqual(0, d0)
        self.assertEqual(1, d1)
        self.assertEqual(0, d2)
        self.assertEqual(d0, d2)

    @skipIfRocm
    @unittest.skipIf(not TEST_MULTIGPU, 'detected only one GPU')
    @unittest.skipIf(not TEST_LARGE_TENSOR, 'not enough memory')
    @skipCUDANonDefaultStreamIf(True)
    def test_streams_and_events(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def test_default_streams_with_device_index_args():
            if False:
                print('Hello World!')
            s0 = torch.cuda.default_stream(0)
            s1 = torch.cuda.default_stream(1)
            return (s0.device_index(), s1.device_index())
        (d0, d1) = test_default_streams_with_device_index_args()
        self.assertEqual(d0, 0)
        self.assertEqual(d1, 1)

        @torch.jit.script
        def test_default_streams():
            if False:
                while True:
                    i = 10
            s0 = torch.cuda.default_stream(torch.device('cuda:0'))
            s1 = torch.cuda.default_stream(torch.device('cuda:1'))
            d = torch.device('cuda:1')
            s2 = torch.cuda.current_stream(torch.device('cuda:0'))
            check_s2 = s2.id() == s0.id()
            check_d0 = torch.cuda.current_device() == s2.device_index()
            with torch.cuda.device(d):
                s3 = torch.cuda.current_stream(d)
                check_s3 = s3.id() == s1.id()
                check_d1 = torch.cuda.current_device() == s3.device_index()
            is_device_d0 = torch.cuda.current_device() == s2.device_index()
            return (s0.device_index(), s1.device_index(), check_s2, check_s3, check_d0, check_d1, is_device_d0)
        (d0, d1, check_s2, check_s3, check_d0, check_d1, is_device_d0) = test_default_streams()
        self.assertEqual(d0, 0)
        self.assertEqual(d1, 1)
        self.assertTrue(check_s2)
        self.assertTrue(check_s3)
        self.assertTrue(check_d0)
        self.assertTrue(check_d1)
        self.assertTrue(is_device_d0)

        @torch.jit.script
        def test_set_none_stream():
            if False:
                i = 10
                return i + 15
            device_index = torch.cuda.current_device()
            device = torch.device('cuda:' + str(device_index))
            current_stream = torch.cuda.current_stream(device)
            default_stream = torch.cuda.default_stream(device)
            with torch.cuda.stream(None):
                cur_device_index = torch.cuda.current_device()
                is_device_index_same = cur_device_index == device_index
                is_current_stream_same = torch.cuda.current_stream(device).id() == current_stream.id()
                is_default_stream_same = torch.cuda.default_stream(device).id() == default_stream.id()
            are_streams_same = is_device_index_same and is_current_stream_same and is_default_stream_same
            return are_streams_same
        self.assertTrue(test_set_none_stream())

        @torch.jit.script
        def test_set_device_none():
            if False:
                for i in range(10):
                    print('nop')
            device_index = torch.cuda.current_device()
            with torch.cuda.device(None):
                is_device_same = torch.cuda.current_device() == device_index
            return is_device_same
        self.assertTrue(test_set_device_none())

        @torch.jit.script
        def test_simple_stream():
            if False:
                print('Hello World!')
            device_index = torch.cuda.current_device()
            s = torch.cuda.Stream()
            return device_index == s.device_index()
        self.assertTrue(test_simple_stream(), 'Could not create Stream!')

        class Result(NamedTuple):
            t1: torch.Tensor
            t2: torch.Tensor
            is_current_and_default_stream_same: bool
            is_default_and_user_stream_not_same: bool
            is_stream_set: bool
            is_stream_reset: bool
            default_stream_query: bool
            default_stream_id: int
            user_stream_id: int

        @torch.jit.script
        def test_get_stream():
            if False:
                for i in range(10):
                    print('nop')
            device_index = torch.cuda.current_device()
            device = torch.device('cuda:' + str(device_index))
            current_stream = torch.cuda.current_stream(device)
            default_stream = torch.cuda.default_stream(device)
            user_stream = torch.cuda.Stream()
            is_current_and_default_stream_same = current_stream.id() == default_stream.id()
            is_default_and_user_stream_not_same = default_stream.id() != user_stream.id()
            with torch.cuda.stream(user_stream):
                is_stream_set = torch.cuda.current_stream(device).id() == user_stream.id()
            is_stream_reset = torch.cuda.current_stream(device).id() == current_stream.id()
            tensor1 = torch.rand(10000, 10000, device='cuda')
            tensor2 = torch.mm(tensor1, tensor1).to('cuda')
            default_stream.synchronize()
            default_stream_query = default_stream.query()
            res = Result(tensor1, tensor2, is_current_and_default_stream_same, is_default_and_user_stream_not_same, is_stream_set, is_stream_reset, default_stream_query, default_stream.id(), user_stream.id())
            return res
        result = test_get_stream()
        self.assertEqual(torch.matmul(result.t1, result.t1), result.t2)
        self.assertTrue(result.is_current_and_default_stream_same)
        self.assertTrue(result.is_default_and_user_stream_not_same)
        self.assertTrue(result.is_stream_set)
        self.assertTrue(result.is_stream_reset)
        self.assertTrue(result.default_stream_query)
        self.assertEqual(result.default_stream_id, 0)
        self.assertNotEqual(result.user_stream_id, 0)

        @torch.jit.script
        def test_stream_context():
            if False:
                for i in range(10):
                    print('nop')
            device_index = torch.cuda.current_device()
            device = torch.device('cuda:' + str(device_index))
            current_stream = torch.cuda.current_stream(device)
            user_stream = torch.cuda.Stream()
            A = torch.rand(1000, 1000, device='cuda')
            with torch.cuda.stream(user_stream):
                check = torch.cuda.current_stream(device).id() == user_stream.id()
                B = torch.mm(A, A).to('cuda')
            user_stream.synchronize()
            is_stream_reset = torch.cuda.current_stream(device).id() == current_stream.id()
            return (A, B, check, is_stream_reset)
        (A, B, is_stream_set, is_stream_reset) = test_stream_context()
        self.assertEqual(torch.matmul(A, A), B)
        self.assertTrue(is_stream_set, 'Error: Current stream was not set to user stream!')
        self.assertTrue(is_stream_reset, 'Error: The stream was not restored to previous stream!')

        @torch.jit.script
        def test_multiple_stream():
            if False:
                for i in range(10):
                    print('nop')
            prev_device_index = torch.cuda.current_device()
            device = torch.device('cuda:' + str(prev_device_index))
            prev_current_stream = torch.cuda.current_stream(device)
            d1 = torch.device('cuda:0')
            d2 = torch.device('cuda:1')
            s1 = torch.cuda.Stream(d1, 0)
            s2 = torch.cuda.Stream(d2, 0)
            A = torch.rand(1000, 1000, device='cuda')
            B = torch.rand(1000, 1000, device='cuda')
            with torch.cuda.stream(s1):
                C = torch.mm(A, A).to('cuda')
                is_stream_s1 = torch.cuda.current_stream(d1).id() == s1.id()
                is_device_s1 = torch.cuda.current_device() == s1.device_index()
                with torch.cuda.stream(s2):
                    is_stream_s2 = torch.cuda.current_stream(d2).id() == s2.id()
                    is_device_s2 = torch.cuda.current_device() == s2.device_index()
                    D = torch.mm(B, B).to('cuda')
                is_stream_s1_after = torch.cuda.current_stream(d1).id() == s1.id()
                is_device_s1_after = torch.cuda.current_device() == s1.device_index()
                s2.synchronize()
            s1.synchronize()
            is_device_current = torch.cuda.current_device() == prev_device_index
            is_stream_current = torch.cuda.current_stream(device).id() == prev_current_stream.id()
            check_stream = is_stream_s1 and is_stream_s2 and is_stream_s1_after and is_stream_current
            check_device = is_device_s1 and is_device_s2 and is_device_s1_after and is_device_current
            return (A, B, C, D, check_stream, check_device)
        (A, B, C, D, check_stream, check_device) = test_multiple_stream()
        self.assertEqual(torch.matmul(A, A), C)
        self.assertEqual(torch.matmul(B, B), D)
        self.assertTrue(check_stream)
        self.assertTrue(check_device)

        @torch.jit.script
        def test_data_dependency_between_streams():
            if False:
                i = 10
                return i + 15
            device_index = torch.cuda.current_device()
            device = torch.device('cuda:' + str(device_index))
            prev_current_stream = torch.cuda.current_stream(device)
            d = torch.device('cuda:0')
            s1 = torch.cuda.Stream(d, 0)
            s2 = torch.cuda.Stream(d, 0)
            event = torch.cuda.Event(False, False, False)
            A = torch.rand(1000, 1000, device='cuda')
            with torch.cuda.stream(s1):
                is_stream_s1 = torch.cuda.current_stream(device).id() == s1.id()
                B = torch.mm(A, A).to('cuda')
            s1.record_event(event)
            is_current_stream_1 = torch.cuda.current_stream(device).id() == prev_current_stream.id()
            s2.wait_event(event)
            with torch.cuda.stream(s2):
                is_stream_s2 = torch.cuda.current_stream(device).id() == s2.id()
                C = torch.mm(B, B).to('cuda')
            s2.synchronize()
            is_current_stream_2 = torch.cuda.current_stream(device).id() == prev_current_stream.id()
            check_stream = is_current_stream_1 and is_current_stream_2 and is_stream_s1 and is_stream_s2
            return (A, B, C, check_stream)
        (A, B, C, check_stream) = test_data_dependency_between_streams()
        self.assertEqual(torch.matmul(A, A), B)
        self.assertEqual(torch.matmul(B, B), C)
        self.assertTrue(check_stream)

        @torch.jit.script
        def test_simple_event():
            if False:
                return 10
            e = torch.cuda.Event(True, False, False)
            return e is not None
        self.assertTrue(test_simple_event(), 'Could not create CUDA Event!')

        @torch.jit.script
        def test_event():
            if False:
                return 10
            device_index = torch.cuda.current_device()
            device = torch.device('cuda:' + str(device_index))
            stream = torch.cuda.current_stream(device)
            event = torch.cuda.Event(True, False, False)
            is_true_event_query = event.query()
            start_event = torch.cuda.Event(True, False, False)
            stream.record_event(start_event)
            tensor1 = torch.rand(1000000000, 1000000000, device='cuda')
            tensor2 = torch.mm(tensor1, tensor1).to('cuda')
            stream.record_event(event)
            event.synchronize()
            is_again_true_event_query = event.query()
            if not (is_true_event_query and is_again_true_event_query):
                return -1.0
            return start_event.elapsed_time(event)
        self.assertGreater(test_event(), 0)

        @torch.jit.script
        def test_stream_synchronize() -> float:
            if False:
                for i in range(10):
                    print('nop')
            device_index = torch.cuda.current_device()
            s = torch.cuda.Stream()
            e_tik = torch.cuda.Event(True, False, False)
            e_tok = torch.cuda.Event(True, False, False)
            e_tik.record(s)
            tensor1 = torch.rand(1000000000, 1000000000, device='cuda')
            with torch.cuda.stream(s):
                tensor2 = torch.mm(tensor1, tensor1).to('cuda')
            s.synchronize()
            e_tok.record(s)
            e_tok.synchronize()
            if not s.query():
                return -1.0
            return e_tik.elapsed_time(e_tok)
        self.assertGreater(test_stream_synchronize(), 0)

        @torch.jit.script
        def test_event_synchronize() -> float:
            if False:
                return 10
            s = torch.cuda.Stream()
            e_tik = torch.cuda.Event(True, False, False)
            e_tok = torch.cuda.Event(True, False, False)
            e_tik.record(s)
            tensor1 = torch.rand(1000000000, 1000000000, device='cuda')
            with torch.cuda.stream(s):
                tensor = torch.mm(tensor1, tensor1).to('cuda')
            s.record_event(e_tok)
            e_tok.synchronize()
            s.synchronize()
            if not s.query():
                return -1.0
            return e_tik.elapsed_time(e_tok)
        self.assertGreater(test_event_synchronize(), 0)

        @torch.jit.script
        def test_event_wait() -> float:
            if False:
                i = 10
                return i + 15
            device_index = torch.cuda.current_device()
            device = torch.device('cuda:' + str(device_index))
            s0 = torch.cuda.current_stream(device)
            s1 = torch.cuda.Stream()
            e_tik = torch.cuda.Event(True, True, False)
            e_tok = torch.cuda.Event(True, True, False)
            e_tik.record(s0)
            tensor1 = torch.rand(1000000000, 1000000000, device='cuda')
            with torch.cuda.stream(s0):
                tensor2 = torch.mm(tensor1, tensor1).cuda()
            e_sync = torch.cuda.Event(True, False, False)
            e_sync.record(torch.cuda.current_stream(device))
            e_sync.wait(s1)
            with torch.cuda.stream(s1):
                tensor3 = torch.rand(1000000000, 1000000000, device='cuda')
                tensor4 = torch.mm(tensor3, tensor3).cuda()
            s1.synchronize()
            e_tok.record(torch.cuda.current_stream(device))
            e_tok.synchronize()
            s0.synchronize()
            if not s0.query() or not s1.query() or (not e_sync.query()):
                return -1.0
            return e_tik.elapsed_time(e_tok)
        self.assertGreater(test_event_wait(), 0)

        @torch.jit.script
        def test_wait_event():
            if False:
                for i in range(10):
                    print('nop')
            d1 = torch.device('cuda:1')
            with torch.cuda.device(d1):
                s0 = torch.cuda.current_stream(d1)
                tensor1 = torch.rand(1000000000, 1000000000, device='cuda')
                tensor2 = torch.mm(tensor1, tensor1).to('cuda')
                e0 = torch.cuda.Event(False, False, False)
                s0.record_event(e0)
            s1 = torch.cuda.current_stream(torch.device('cuda:0'))
            s1.wait_event(e0)
            s1.synchronize()
            return e0.query() and s0.query() and s1.query()
        self.assertTrue(test_wait_event())

        def test_save_load(self):
            if False:
                i = 10
                return i + 15

            class Model(torch.nn.Module):

                def forward(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    s = torch.cuda.Stream()
                    a = torch.rand(3, 4, device='cuda')
                    b = torch.rand(3, 4, device='cuda')
                    with torch.cuda.stream(s):
                        is_stream_s = torch.cuda.current_stream(s.device).id() == s.id()
                        c = torch.cat((a, b), 0).cuda()
                    s.synchronize()
                    return (is_stream_s, a, b, c)
            model = Model()
            script_model = torch.jit.script(model)
            (is_stream_s, a, b, c) = script_model()
            self.assertTrue(is_stream_s)
            self.assertEqual(torch.cat((a, b), 0), c)
            load_model = self.getExportImportCopy(script_model)
            (is_stream_s, a_load, b_load, c_load) = load_model()
            self.assertTrue(is_stream_s)
            self.assertEqual(torch.cat((a_load, b_load), 0), c_load)

    @unittest.skipIf(not TEST_CUDA, 'Cuda not available')
    def test__exchange_device_op(self):
        if False:
            while True:
                i = 10

        def fn(device: int, tensor):
            if False:
                while True:
                    i = 10
            torch.cuda._exchange_device(device)
            return tensor.cos().relu()
        fn_s = torch.jit.script(fn)
        g = fn_s.graph
        FileCheck().check('cuda::_exchange_device(').run(g)
        torch._C._jit_pass_inline(g)
        FileCheck().check('cuda::_exchange_device(').run(g)

    @unittest.skipIf(not TEST_CUDA, 'Cuda not available')
    def test__maybe_exchange_device_op(self):
        if False:
            print('Hello World!')

        def fn(device: int, tensor):
            if False:
                for i in range(10):
                    print('nop')
            torch.cuda._maybe_exchange_device(device)
            return tensor.cos().relu()
        fn_s = torch.jit.script(fn)
        g = fn_s.graph
        FileCheck().check('cuda::_maybe_exchange_device(').run(g)
        torch._C._jit_pass_inline(g)
        FileCheck().check('cuda::_maybe_exchange_device(').run(g)