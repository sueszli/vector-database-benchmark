import sys
import textwrap
import traceback
from typing import List
import torch
import torch.cuda._sanitizer as csan
from torch.cuda._sanitizer import StreamId, DataPtr, EventId
from torch.testing._internal.common_utils import TestCase, run_tests, NoTest, TEST_CUDA
if not TEST_CUDA:
    print('CUDA not available, skipping tests', file=sys.stderr)
    TestCase = NoTest

class TestArgumentHandler(TestCase):

    def test_add(self):
        if False:
            i = 10
            return i + 15
        add_func = torch.ops.aten.add.Tensor
        a = torch.ones(5, 3, device='cuda')
        b = torch.randn(5, 3, device='cuda')
        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(add_func._schema, (a, b), {})
        c = torch.add(a, b)
        argument_handler.parse_outputs(c)
        self.assertEqual({a.data_ptr(), b.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual({c.data_ptr()}, argument_handler.dataptrs_written)

    def test_cat(self):
        if False:
            i = 10
            return i + 15
        cat_func = torch.ops.aten.cat.default
        a = torch.ones(2, 4, 5, device='cuda')
        b = torch.zeros(2, 1, 5, device='cuda')
        c = torch.rand(2, 7, 5, device='cuda')
        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(cat_func._schema, ([a, b, c], 1), {})
        d = torch.cat((a, b, c), dim=1)
        argument_handler.parse_outputs(d)
        self.assertEqual({a.data_ptr(), b.data_ptr(), c.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual({d.data_ptr()}, argument_handler.dataptrs_written)

    def test_split(self):
        if False:
            while True:
                i = 10
        split_func = torch.ops.aten.split.Tensor
        a = torch.arange(10, device='cuda').reshape(5, 2)
        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(split_func._schema, (a, 2), {})
        out = torch.split(a, 2)
        argument_handler.parse_outputs(out)
        outputs = {out[0].data_ptr(), out[1].data_ptr(), out[2].data_ptr()}
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual(outputs, argument_handler.dataptrs_written)

    def test_inplace(self):
        if False:
            print('Hello World!')
        add_inplace_func = torch.ops.aten.add_.Tensor
        a = torch.rand(4, 2, device='cuda')
        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(add_inplace_func._schema, (a, 5), {})
        a.add_(5)
        argument_handler.parse_outputs(a)
        self.assertEqual(set(), argument_handler.dataptrs_read)
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_written)

    def test_out(self):
        if False:
            for i in range(10):
                print('nop')
        mul_out_func = torch.ops.aten.mul.out
        a = torch.arange(8, device='cuda')
        b = torch.empty(8, device='cuda')
        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(mul_out_func._schema, (a, 3), {'out': b})
        torch.mul(a, 3, out=b)
        argument_handler.parse_outputs(b)
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual({b.data_ptr()}, argument_handler.dataptrs_written)

    def test_nonzero(self):
        if False:
            return 10
        nonzero_func = torch.ops.aten.nonzero.default
        a = torch.ones(5, 3, 2, device='cuda')
        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(nonzero_func._schema, (a,), {'as_tuple': True})
        out = torch.nonzero(a, as_tuple=True)
        argument_handler.parse_outputs(out)
        outputs = {out[0].data_ptr(), out[1].data_ptr(), out[2].data_ptr()}
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual(outputs, argument_handler.dataptrs_written)

    def test_tensor_names(self):
        if False:
            i = 10
            return i + 15
        addr_func = torch.ops.aten.addr.default
        vec = torch.arange(1, 4, device='cuda')
        M = torch.zeros(3, 3, device='cuda')
        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(addr_func._schema, (M, vec, vec), {})
        out = torch.addr(M, vec, vec)
        argument_handler.parse_outputs(out)
        self.assertEqual(argument_handler.tensor_aliases, {M.data_ptr(): ['self'], vec.data_ptr(): ['vec1', 'vec2'], out.data_ptr(): []})
        self.assertEqual({out.data_ptr()}, argument_handler.outputs)

def tensor_id(i: int) -> DataPtr:
    if False:
        for i in range(10):
            print('nop')
    return i

def stream_id(i: int) -> StreamId:
    if False:
        for i in range(10):
            print('nop')
    return 1000 + i

def event_id(i: int) -> EventId:
    if False:
        i = 10
        return i + 15
    return 2000 + i

class TestEventHandler(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.handler = csan.EventHandler()

    def kernel_launch(self, stream: StreamId, read_only: List[DataPtr]=None, read_write: List[DataPtr]=None) -> List[csan.SynchronizationError]:
        if False:
            for i in range(10):
                print('nop')
        if read_only is None:
            read_only = []
        if read_write is None:
            read_write = []
        return self.handler._handle_kernel_launch(stream, read_only, read_write, {}, '', {k: [''] for k in read_only + read_write})

    def assert_good_kernel_launch(self, stream: StreamId, read_only: List[DataPtr]=None, read_write: List[DataPtr]=None) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(self.kernel_launch(stream, read_only, read_write), [])

    def assert_bad_kernel_launch(self, number_of_errors: int, stream: StreamId, read_only: List[DataPtr]=None, read_write: List[DataPtr]=None) -> None:
        if False:
            while True:
                i = 10
        errors = self.kernel_launch(stream, read_only, read_write)
        self.assertEqual(len(errors), number_of_errors)

    def test_empty_kernel_launch(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_good_kernel_launch(stream_id(0))

    def test_simple_passing(self):
        if False:
            i = 10
            return i + 15
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])

    def test_simple_error(self):
        if False:
            print('Hello World!')
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_simple_sync(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])

    def test_reads_check_last_write(self):
        if False:
            return 10
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(3), read_only=[tensor_id(1)])

    def test_branch_sync(self):
        if False:
            return 10
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.handler._handle_event_wait(event_id(0), stream_id(3))
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_chain_sync(self):
        if False:
            return 10
        iterations = 10
        self.assert_good_kernel_launch(stream_id(0), read_only=[tensor_id(1)])
        for i in range(iterations):
            self.handler._handle_event_record(event_id(i), stream_id(i))
            self.handler._handle_event_wait(event_id(i), stream_id(i + 1))
        self.assert_good_kernel_launch(stream_id(iterations), read_write=[tensor_id(1)])

    def test_expired_record(self):
        if False:
            print('Hello World!')
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_deleted_record(self):
        if False:
            while True:
                i = 10
        for (should_delete, should_create) in [(True, True), (True, False), (False, True)]:
            self.setUp()
            with self.subTest(should_delete=should_delete, should_create=should_create):
                self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
                self.handler._handle_event_record(event_id(0), stream_id(1))
                if should_delete:
                    self.handler._handle_event_deletion(event_id(0))
                if should_create:
                    self.handler._handle_event_creation(event_id(0))
                self.handler._handle_event_wait(event_id(0), stream_id(2))
                self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_all_reads_checked_failing(self):
        if False:
            i = 10
            return i + 15
        iterations = 10
        for i in range(1, iterations):
            self.assert_good_kernel_launch(stream_id(i), read_only=[tensor_id(1)])
            self.handler._handle_event_record(event_id(i), stream_id(i))
        for i in range(1, iterations):
            self.handler._handle_event_wait(event_id(i), stream_id(0))
        self.assert_good_kernel_launch(stream_id(iterations), read_only=[tensor_id(1)])
        self.handler._handle_event_record(event_id(iterations), stream_id(i))
        self.assert_bad_kernel_launch(1, stream_id(0), read_write=[tensor_id(1)])

    def test_all_reads_checked_passing(self):
        if False:
            print('Hello World!')
        iterations = 10
        for i in range(1, iterations):
            self.assert_good_kernel_launch(stream_id(i), read_only=[tensor_id(1)])
            self.handler._handle_event_record(event_id(i), stream_id(i))
        for i in range(1, iterations):
            self.handler._handle_event_wait(event_id(i), stream_id(0))
        self.assert_good_kernel_launch(stream_id(0), read_write=[tensor_id(1)])

    def test_multiple_errors(self):
        if False:
            for i in range(10):
                print('nop')
        iterations = 10
        self.assert_good_kernel_launch(stream_id(0), read_write=[tensor_id(i) for i in range(iterations)])
        self.assert_bad_kernel_launch(iterations, stream_id(1), read_write=[tensor_id(i) for i in range(iterations)])

    def test_correct_state_merging(self):
        if False:
            print('Hello World!')
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(2)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.handler._handle_event_record(event_id(2), stream_id(2))
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(2)])
        self.handler._handle_event_wait(event_id(1), stream_id(2))
        self.handler._handle_event_wait(event_id(2), stream_id(1))
        self.handler._handle_event_record(event_id(3), stream_id(2))
        self.handler._handle_event_wait(event_id(3), stream_id(1))
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1), tensor_id(2)])

    def test_record_override(self):
        if False:
            print('Hello World!')
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(2)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.handler._handle_event_record(event_id(1), stream_id(2))
        self.handler._handle_event_wait(event_id(1), stream_id(3))
        self.assert_bad_kernel_launch(1, stream_id(3), read_write=[tensor_id(1)])

    def test_multiple_wait(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.handler._handle_event_wait(event_id(1), stream_id(2))
        self.handler._handle_event_wait(event_id(1), stream_id(3))
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])

    def test_device_synchronize(self):
        if False:
            print('Hello World!')
        iterations = 10
        for i in range(1, iterations):
            self.assert_good_kernel_launch(stream_id(i), read_write=[tensor_id(i)])
        self.handler._handle_device_synchronization()
        self.assert_good_kernel_launch(stream_id(0), read_write=[tensor_id(i) for i in range(1, iterations)])

    def test_device_synchronization_expired(self):
        if False:
            print('Hello World!')
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_device_synchronization()
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_new_stream_is_synchronized(self):
        if False:
            print('Hello World!')
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_device_synchronization()
        self.handler._handle_stream_creation(stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])

    def test_stream_synchronize(self):
        if False:
            while True:
                i = 10
        self.assert_good_kernel_launch(stream_id(0), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(2)])
        self.handler._handle_stream_synchronization(stream_id(0))
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(4), read_only=[tensor_id(2)])

    def test_event_synchronize(self):
        if False:
            i = 10
            return i + 15
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(2)])
        self.handler._handle_event_synchronization(event_id(1))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(2)])

class TestMessages(TestCase):

    def setUp(self):
        if False:
            return 10
        self.handler = csan.EventHandler()

    def test_ensure_exists(self):
        if False:
            i = 10
            return i + 15
        ARG = 0
        for (func, out) in [(self.handler._handle_event_deletion, f'Found Event with id: {ARG}, but no matching event creation in the trace. Backfilling the trace now. Perhaps the sanitizer was enabled after some torch operations?'), (self.handler._handle_memory_deallocation, f'Found tensor with pointer: {ARG}, but no matching tensor allocation in the trace. Backfilling the trace now. Perhaps the sanitizer was enabled after some torch operations?')]:
            with self.subTest(func=func, out=out):
                with self.assertLogs() as captured:
                    func(ARG)
                self.assertEqual(captured.records[0].getMessage(), out)

    def test_ensure_does_not_exist(self):
        if False:
            for i in range(10):
                print('nop')
        ARG = 0
        self.handler._handle_event_creation(ARG)
        self.handler._handle_stream_creation(ARG)
        for (func, out) in [(self.handler._handle_event_creation, f"Found duplicate event creation in the trace for event with id: {ARG}. Assuming the trace for event deletion wasn't caught and backfilling it now. Perhaps the sanitizer was enabled after some torch operations?"), (self.handler._handle_stream_creation, f'Found duplicate Stream creation in the trace for Stream with id: {ARG}. PyTorch Streams are only created once, so this trace entry is ignored.')]:
            with self.subTest(func=func, out=out):
                with self.assertLogs() as captured:
                    func(ARG)
                self.assertEqual(captured.records[0].getMessage(), out)

    def test_error_message(self):
        if False:
            while True:
                i = 10
        current_access = csan.Access(type=csan.AccessType.WRITE, seq_num=1, stream=stream_id(1), operator='schema', aliases=['b'], is_output=True, stack_trace=traceback.StackSummary.from_list([('file', 0, 'name', 'trace a')]))
        previous_access = csan.Access(type=csan.AccessType.READ, seq_num=2, stream=stream_id(0), operator='schema', aliases=['a'], is_output=False, stack_trace=traceback.StackSummary.from_list([('file', 0, 'name', 'trace b')]))
        error = csan.UnsynchronizedAccessError(data_ptr=tensor_id(1), allocation_stack_trace=traceback.StackSummary.from_list([('file', 0, 'name', 'alloc')]), current_access=current_access, previous_access=previous_access)
        self.assertEqual(str(error), textwrap.dedent('                ============================\n                CSAN detected a possible data race on tensor with data pointer 1\n                Access by stream 1001 during kernel:\n                schema\n                writing to argument(s) b, and to the output\n                With stack trace:\n                  File "file", line 0, in name\n                    trace a\n\n                Previous access by stream 1000 during kernel:\n                schema\n                reading from argument(s) a\n                With stack trace:\n                  File "file", line 0, in name\n                    trace b\n\n                Tensor was allocated with stack trace:\n                  File "file", line 0, in name\n                    alloc\n                '))
if __name__ == '__main__':
    run_tests()