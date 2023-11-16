import multiprocessing
import socket
import time
import unittest
from contextlib import closing
import paddle
from paddle import base
port_set = set()
paddle.enable_static()

class CollectiveCPUBarrierWithGlooTest(unittest.TestCase):

    def find_free_port(self):
        if False:
            while True:
                i = 10

        def _free_port():
            if False:
                return 10
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.bind(('', 0))
                return s.getsockname()[1]
        while True:
            port = _free_port()
            if port not in port_set:
                port_set.add(port)
                return port

    def barrier_func(self, id, rank_num, server_endpoint, out_dict, sleep_time):
        if False:
            i = 10
            return i + 15
        try:
            paddle.distributed.gloo_init_parallel_env(id, rank_num, server_endpoint)
            paddle.distributed.gloo_barrier()
            start = time.time()
            if id == 0:
                time.sleep(sleep_time)
            paddle.distributed.gloo_barrier()
            end = time.time()
            out_dict[id] = end - start
            paddle.distributed.gloo_release()
        except:
            out_dict[id] = 0

    def barrier_op(self, id, rank_num, server_endpoint, out_dict, sleep_time):
        if False:
            print('Hello World!')
        try:
            main_prog = base.Program()
            startup_prog = base.Program()
            paddle.distributed.gloo_init_parallel_env(id, rank_num, server_endpoint)
            place = base.CPUPlace()
            with base.program_guard(main_prog, startup_prog):
                paddle.distributed.barrier()
            exe = base.Executor(place)
            exe.run(main_prog)
            start = time.time()
            if id == 0:
                time.sleep(sleep_time)
            exe.run(main_prog)
            end = time.time()
            out_dict[id] = end - start
            paddle.distributed.gloo_release()
        except:
            out_dict[id] = 0

    def test_barrier_func_with_multiprocess(self):
        if False:
            return 10
        num_of_ranks = 4
        sleep_time = 1
        ep_str = '127.0.0.1:%s' % self.find_free_port()
        manager = multiprocessing.Manager()
        procs_out_dict = manager.dict()
        jobs = []
        for id in range(num_of_ranks):
            p = multiprocessing.Process(target=self.barrier_func, args=(id, num_of_ranks, ep_str, procs_out_dict, sleep_time))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        for (_, v) in procs_out_dict.items():
            self.assertTrue(v > sleep_time)

    def test_barrier_op_with_multiprocess(self):
        if False:
            while True:
                i = 10
        num_of_ranks = 4
        sleep_time = 1
        ep_str = '127.0.0.1:%s' % self.find_free_port()
        manager = multiprocessing.Manager()
        procs_out_dict = manager.dict()
        jobs = []
        for id in range(num_of_ranks):
            p = multiprocessing.Process(target=self.barrier_op, args=(id, num_of_ranks, ep_str, procs_out_dict, sleep_time))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        for (_, v) in procs_out_dict.items():
            self.assertTrue(v > sleep_time)
if __name__ == '__main__':
    unittest.main()