from legacy_test.test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main
import paddle
import paddle.distributed as dist
from paddle import base, framework
from paddle.base import data_feeder
paddle.enable_static()

def alltoall_new(in_tensor_or_tensor_list, out_tensor_or_tensor_list, group=None, sync_op=True):
    if False:
        print('Hello World!')
    op_type = 'all_to_all'
    ring_id = 0 if group is None else group.id
    nranks = dist.get_world_size()
    helper = framework.LayerHelper(op_type, **locals())
    in_tensor = in_tensor_or_tensor_list
    if isinstance(in_tensor_or_tensor_list, list):
        if len(in_tensor_or_tensor_list) == 0:
            raise RuntimeError('The input tensor_list should not be empty.')
        if len(in_tensor_or_tensor_list[0].shape) == 0:
            in_tensor = paddle.stack(in_tensor_or_tensor_list, axis=0)
        else:
            in_tensor = paddle.concat(in_tensor_or_tensor_list, axis=0)
    out_tensor = out_tensor_or_tensor_list
    if isinstance(out_tensor_or_tensor_list, list):
        if len(out_tensor_or_tensor_list) != 0:
            raise ValueError("The 'out_tensor_list' for all_to_all must be an empty list.")
        out_tensor = helper.create_variable_for_type_inference(dtype=in_tensor.dtype)
    data_feeder.check_variable_and_dtype(in_tensor, 'in_tensor', ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool', 'uint16'], 'all_to_all')
    helper.append_op(type=op_type, inputs={'x': [in_tensor]}, outputs={'out': [out_tensor]}, attrs={'ring_id': ring_id})
    if isinstance(out_tensor_or_tensor_list, list):
        if not sync_op:
            dist.wait(out_tensor, use_calc_stream=False)
        if len(in_tensor_or_tensor_list[0].shape) == 0:
            out_tensor_or_tensor_list.extend(paddle.unstack(out_tensor, 0))
        else:
            out_tensor_or_tensor_list.extend(paddle.split(out_tensor, nranks, 0))

class TestCollectiveAllToAllAPI(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            print('Hello World!')
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[-1, 10, 1000], dtype='float32')
            tindata.desc.set_need_check_feed(False)
            tindata = paddle.split(tindata, 2, axis=0)
            tout_data = []
            paddle.distributed.alltoall(tindata, tout_data)
            return tout_data

    def get_model_new(self, main_prog, startup_program, rank, dtype=None, reduce_type=None):
        if False:
            i = 10
            return i + 15
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[-1, 10, 1000], dtype=dtype)
            tindata.desc.set_need_check_feed(False)
            tindata = paddle.split(tindata, 2, axis=0)
            tout_data = []
            alltoall_new(tindata, tout_data)
            return tout_data

    def get_model_new_comm(self, main_prog, startup_program, rank, dtype='float32'):
        if False:
            return 10
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[-1, 10, 1000], dtype=dtype)
            tindata.desc.set_need_check_feed(False)
            tindata = paddle.split(tindata, 2, axis=0)
            tout_data = []
            paddle.distributed.alltoall(tindata, tout_data)
            return tout_data
if __name__ == '__main__':
    runtime_main(TestCollectiveAllToAllAPI, 'alltoall')