import os
import random
import sys
import unittest
import numpy as np
import paddle
import paddle.static
from ..op_test_ipu import IPUOpTest
mpi_comm = None

@unittest.skip('Disable distributed tests on auto CI.')
class TestBase(IPUOpTest):

    def set_attrs(self, enable_ipu, optimizer, log, onchip=False, rts=False):
        if False:
            for i in range(10):
                print('nop')
        self.ipu_options = {'enable_pipelining': True, 'batches_per_step': 1, 'enable_gradient_accumulation': True, 'accumulation_factor': 4, 'enable_replicated_graphs': True, 'replicated_graph_count': 2, 'location_optimizer': {'on_chip': onchip, 'use_replicated_tensor_sharding': rts}}
        self.cpu_bs = 16
        self.ipu_bs = 1
        self.optimizer = optimizer
        self.log = log
        self.enable_ipu = enable_ipu

    def test(self):
        if False:
            while True:
                i = 10
        seed = 2021
        np.random.seed(seed)
        random.seed(seed)
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = seed
        startup_prog.random_seed = seed
        bs = self.ipu_bs if self.enable_ipu else self.cpu_bs
        data = np.random.rand(1, 3, 10, 10).astype(np.float32)
        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                image = paddle.static.data(name='image', shape=[bs, 3, 10, 10], dtype='float32')
                with paddle.static.ipu_shard_guard(index=0, stage=0):
                    conv1 = paddle.static.nn.conv2d(image, num_filters=3, filter_size=3, bias_attr=False)
                with paddle.static.ipu_shard_guard(index=1, stage=1):
                    conv2 = paddle.static.nn.conv2d(conv1, num_filters=3, filter_size=3, bias_attr=False)
                    loss = paddle.mean(conv2)
                if self.optimizer == 'sgd':
                    opt = paddle.optimizer.SGD(learning_rate=0.01)
                elif self.optimizer == 'adam':
                    opt = paddle.optimizer.Adam(learning_rate=0.01)
                elif self.optimizer == 'lamb':
                    opt = paddle.optimizer.Lamb(learning_rate=0.01)
                else:
                    raise Exception('optimizer must be sgd, adam or lamb')
                opt.minimize(loss)
                if self.enable_ipu:
                    place = paddle.IPUPlace()
                else:
                    place = paddle.CPUPlace()
                executor = paddle.static.Executor(place)
                executor.run(startup_prog)
                if self.enable_ipu:
                    feed_list = [image.name]
                    fetch_list = [loss.name]
                    ipu_strategy = paddle.static.IpuStrategy()
                    ipu_strategy.set_graph_config(num_ipus=2 * self.ipu_options['replicated_graph_count'], is_training=True, enable_manual_shard=True)
                    ipu_strategy.set_options(self.ipu_options)
                    ipu_strategy.set_options({'enable_distribution': True, 'enable_distributed_replicated_graphs': True, 'global_replica_offset': int(os.environ.get('PADDLE_TRAINER_ID')) * 2, 'global_replication_factor': 4})
                    program = paddle.static.IpuCompiledProgram(main_prog, ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
                    feed = {'image': np.tile(data, [self.ipu_options['replicated_graph_count'] * self.ipu_options['batches_per_step'] * self.ipu_options['accumulation_factor'], 1, 1, 1])}
                else:
                    program = main_prog
                    feed = {'image': np.tile(data, [self.cpu_bs, 1, 1, 1])}
                epoch = 10
                if not self.enable_ipu:
                    epoch *= 4
                    epoch *= self.ipu_options['batches_per_step']
                    epoch *= self.ipu_options['accumulation_factor']
                    epoch = epoch / (self.cpu_bs / self.ipu_bs)
                results = []
                for i in range(int(epoch)):
                    res = executor.run(program, feed=feed, fetch_list=[loss])
                    if self.enable_ipu:
                        res = mpi_comm.gather(res, root=0)
                    results.append(res)
                if self.enable_ipu:
                    if int(os.environ.get('PADDLE_TRAINER_ID')) == 0:
                        np.savetxt(self.log, np.array(results).flatten())
                else:
                    np.savetxt(self.log, np.array(results).flatten())
if __name__ == '__main__':
    paddle.enable_static()
    if len(sys.argv) == 5:
        from mpi4py import MPI
        DISTRIBUTED_COMM = MPI.COMM_WORLD

        def _get_comm():
            if False:
                i = 10
                return i + 15
            global DISTRIBUTED_COMM
            if DISTRIBUTED_COMM is None:
                raise RuntimeError('Distributed Commumication not setup. Please run setup_comm(MPI.COMM_WORLD) first.')
            return DISTRIBUTED_COMM
        mpi_comm = _get_comm()
        optimizer = sys.argv[1]
        log = sys.argv[2]
        onchip = True if sys.argv[3] == 'True' else False
        rts = True if sys.argv[4] == 'True' else False
        test = TestBase()
        test.set_attrs(enable_ipu=True, optimizer=optimizer, log=log, onchip=onchip, rts=rts)
        test.test()
    elif len(sys.argv) == 3:
        test = TestBase()
        test.set_attrs(enable_ipu=False, optimizer=sys.argv[1], log=sys.argv[2])
        test.test()
    else:
        raise ValueError('Only support 3 or 5 args. 3 for cpu test, 5 for ipu distributed test')