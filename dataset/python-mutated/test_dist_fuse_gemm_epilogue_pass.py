import unittest
import numpy as np
from dist_pass_test_base import DistPassTestBase
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.passes import PassManager, new_pass
paddle.enable_static()
np.random.seed(12345)
paddle.seed(12345)

def verify_op_count(op_types, op_name, target_count):
    if False:
        i = 10
        return i + 15
    count = 0
    for op_type in op_types:
        if op_type == op_name:
            count += 1
    return count == target_count

class MultiFCLayer(nn.Layer):

    def __init__(self, hidden, Activation):
        if False:
            print('Hello World!')
        super().__init__()
        self.linear1 = paddle.nn.Linear(hidden, 4 * hidden)
        self.linear2 = paddle.nn.Linear(4 * hidden, hidden)
        self.linear3 = paddle.nn.Linear(hidden, hidden)
        self.relu1 = Activation()
        self.relu2 = Activation()
        self.relu3 = Activation()

    def forward(self, x, matmul_y, ele_y):
        if False:
            for i in range(10):
                print('nop')
        output = self.linear1(x)
        output = self.relu1(output)
        output = self.linear2(output)
        output1 = paddle.matmul(output, matmul_y)
        output = self.linear3(output)
        output = self.relu2(output)
        output = paddle.matmul(output, matmul_y)
        output = paddle.add(output, ele_y)
        output = self.relu3(output)
        output = paddle.add(output, output1)
        return output

class TestFuseGemmEpiloguePassReluFP32(DistPassTestBase):

    def init(self):
        if False:
            print('Hello World!')
        self.atol = 0.001
        self.rtol = 0.001
        self.activation = nn.ReLU
        self.act_fwd_name = 'relu'
        self.act_bwd_name = 'relu_grad'
        self.batch = 64
        self.seqlen = 128
        self.hidden = 768
        self.precision = 'FP32'

    def get_model(self, place):
        if False:
            i = 10
            return i + 15
        data = paddle.static.data(name='_data', shape=[-1, self.seqlen, self.hidden], dtype='float32')
        matmul_y = paddle.static.data(name='_matmul_y', shape=[1, self.hidden, self.hidden], dtype='float32')
        ele_y = paddle.static.data(name='_ele_y', shape=[self.hidden], dtype='float32')
        model = MultiFCLayer(self.hidden, self.activation)
        out = model(data, matmul_y, ele_y)
        loss = paddle.mean(out)
        optimizer = paddle.optimizer.Adam(learning_rate=0.001)
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.fuse_all_reduce_ops = False
        dist_strategy.without_graph_optimization = True
        if self.precision == 'AMP':
            dist_strategy.amp = True
            dist_strategy.amp_configs = {'init_loss_scaling': 32768, 'use_dynamic_loss_scaling': True, 'custom_white_list': ['gelu']}
        fleet.init(is_collective=True, strategy=dist_strategy)
        optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(loss)
        rank = paddle.distributed.get_rank()

        def reader():
            if False:
                i = 10
                return i + 15
            for _ in range(10):
                data_arr = np.random.random((self.batch, self.seqlen, self.hidden)).astype('float32') - 0.5
                matmul_y_arr = np.random.random((1, self.hidden, self.hidden)).astype('float32') - 0.5
                ele_y_arr = np.random.random((self.hidden,)).astype('float32') - 0.5
                yield [data_arr, matmul_y_arr, ele_y_arr]
        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        fetch_list = []
        for p in model.parameters():
            grad_name = p.name + '@GRAD'
            fetch_list.append(grad_name)
        fetch_list.append(loss.name)
        return (main_program, startup_program, [data, matmul_y, ele_y], fetch_list, reader)

    def apply_passes(self, main_prog, startup_prog):
        if False:
            i = 10
            return i + 15
        pass_manager = PassManager([new_pass('fuse_gemm_epilogue')])
        pass_manager.apply([main_prog], [startup_prog])
        print(pass_manager.names)
        op_type = []
        for op in main_prog.global_block().ops:
            op_type.append(op.type)
        print(op_type)
        self.assertTrue(verify_op_count(op_type, 'fused_gemm_epilogue', 3))
        self.assertTrue(verify_op_count(op_type, 'fused_gemm_epilogue_grad', 3))
        self.assertTrue(verify_op_count(op_type, self.act_fwd_name, 1))
        self.assertTrue(verify_op_count(op_type, self.act_bwd_name, 2))

    def test_fuse_gemm_epilogue(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_main()

class TestFuseGemmEpiloguePassReluFP16(TestFuseGemmEpiloguePassReluFP32):

    def init(self):
        if False:
            i = 10
            return i + 15
        self.atol = 0.001
        self.rtol = 0.001
        self.activation = nn.ReLU
        self.act_fwd_name = 'relu'
        self.act_bwd_name = 'relu_grad'
        self.batch = 64
        self.seqlen = 128
        self.hidden = 768
        self.precision = 'AMP'

class TestFuseGemmEpiloguePassGeluFP32(TestFuseGemmEpiloguePassReluFP32):

    def init(self):
        if False:
            return 10
        self.atol = 0.001
        self.rtol = 0.001
        self.activation = nn.GELU
        self.act_fwd_name = 'gelu'
        self.act_bwd_name = 'gelu_grad'
        self.batch = 64
        self.seqlen = 128
        self.hidden = 768
        self.precision = 'FP32'

class TestFuseGemmEpiloguePassGeluFP16(TestFuseGemmEpiloguePassReluFP32):

    def init(self):
        if False:
            return 10
        self.atol = 0.005
        self.rtol = 0.001
        self.activation = nn.GELU
        self.act_fwd_name = 'gelu'
        self.act_bwd_name = 'gelu_grad'
        self.batch = 64
        self.seqlen = 128
        self.hidden = 768
        self.precision = 'AMP'
if __name__ == '__main__':
    unittest.main()