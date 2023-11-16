import os
import signal
import time
import unittest
from multiprocessing import Process
import numpy as np
from dist_test_utils import remove_ps_flag
import paddle
from paddle import base
from paddle.base import core
from paddle.base.layers import ops
from paddle.incubate.nn.layer.io import ListenAndServ, Recv, Send
RPC_OP_ROLE_ATTR_NAME = op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC

class TestSendOp(unittest.TestCase):

    def test_send(self):
        if False:
            while True:
                i = 10
        remove_ps_flag(os.getpid())
        place = base.CPUPlace()
        p = Process(target=self.init_serv, args=(place,))
        p.daemon = True
        p.start()
        self.ps_timeout = 5
        self._wait_ps_ready(p.pid)
        with open('/tmp/paddle.%d.port' % p.pid, 'r') as fn:
            selected_port = int(fn.readlines()[0])
        self.init_client(place, selected_port)
        self.run_local(place)
        np.testing.assert_allclose(self.local_out, self.dist_out, rtol=1e-05)
        os.kill(p.pid, signal.SIGINT)
        p.join()

    def _wait_ps_ready(self, pid):
        if False:
            for i in range(10):
                print('nop')
        start_left_time = self.ps_timeout
        sleep_time = 0.5
        while True:
            assert start_left_time >= 0, 'wait ps ready failed'
            time.sleep(sleep_time)
            try:
                os.stat('/tmp/paddle.%d.port' % pid)
                return
            except OSError:
                start_left_time -= sleep_time

    def init_serv(self, place):
        if False:
            print('Hello World!')
        main = base.Program()
        with base.program_guard(main):
            serv = ListenAndServ('127.0.0.1:0', ['X'], optimizer_mode=False)
            with serv.do():
                out_var = main.global_block().create_var(name='scale_0.tmp_0', psersistable=True, dtype='float32', shape=[32, 32])
                x = paddle.static.data(shape=[32, 32], dtype='float32', name='X')
                paddle.nn.initializer.Constant(value=1.0)(x, main.global_block())
                ops._scale(x=x, scale=10.0, out=out_var)
        self.server_exe = base.Executor(place)
        self.server_exe.run(main)

    def init_client(self, place, port):
        if False:
            return 10
        main = base.Program()
        with base.program_guard(main):
            main.global_block().append_op(type='fetch_barrier', inputs={}, outputs={'Out': []}, attrs={'endpoints': [f'127.0.0.1:{port}'], RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE})
            x = paddle.static.data(shape=[32, 32], dtype='float32', name='X')
            x.persistable = True
            paddle.nn.initializer.Constant(value=2.3)(x, main.global_block())
            get_var = main.global_block().create_var(name='scale_0.tmp_0', dtype='float32', persistable=False, shape=[32, 32])
            paddle.nn.initializer.Constant(value=2.3)(get_var, main.global_block())
            Send('127.0.0.1:%d' % port, [x])
            o = Recv('127.0.0.1:%d' % port, [get_var])
        exe = base.Executor(place)
        self.dist_out = exe.run(main, fetch_list=o)

    def run_local(self, place):
        if False:
            return 10
        main = base.Program()
        with base.program_guard(main):
            x = paddle.static.data(shape=[32, 32], dtype='float32', name='X')
            paddle.nn.initializer.Constant(value=2.3)(x, main.global_block())
            o = paddle.scale(x=x, scale=10.0)
        exe = base.Executor(place)
        self.local_out = exe.run(main, fetch_list=[o])
if __name__ == '__main__':
    unittest.main()