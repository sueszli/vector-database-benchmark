import os
import unittest
import paddle
from paddle import base
paddle.enable_static()

class TestCCommInitOp(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.endpoints = os.getenv('PADDLE_TRAINER_ENDPOINTS').split(',')
        self.current_endpoint = os.getenv('PADDLE_CURRENT_ENDPOINT')
        self.nranks = len(self.endpoints)
        self.rank = self.endpoints.index(self.current_endpoint)
        self.gpu_id = int(os.getenv('FLAGS_selected_gpus'))
        self.place = base.CUDAPlace(self.gpu_id)
        self.exe = base.Executor(self.place)
        self.endpoints.remove(self.current_endpoint)
        self.other_endpoints = self.endpoints

    def test_specifying_devices(self):
        if False:
            return 10
        program = base.Program()
        block = program.global_block()
        nccl_id_var = block.create_var(name=base.unique_name.generate('nccl_id'), persistable=True, type=base.core.VarDesc.VarType.RAW)
        block.append_op(type='c_gen_nccl_id', inputs={}, outputs={'Out': nccl_id_var}, attrs={'rank': self.rank, 'endpoint': self.current_endpoint, 'other_endpoints': self.other_endpoints})
        block.append_op(type='c_comm_init', inputs={'X': nccl_id_var}, outputs={}, attrs={'nranks': self.nranks, 'rank': self.rank, 'ring_id': 0, 'device_id': self.gpu_id})
        self.exe.run(program)
if __name__ == '__main__':
    unittest.main()