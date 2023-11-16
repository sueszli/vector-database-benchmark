from test_collective_multi_nodes import TestCollectiveAPIRunnerBase, runtime_main

class TestDygrapgHybridDP(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def check_pass(self, *args, **kwargs):
        if False:
            return 10
        from common import init_parallel_env
        import paddle
        hcg = init_parallel_env('DP16-MP1-PP1-SH1-O1', 2)
        import numpy as np
        dp_group = hcg.get_data_parallel_group()
        np.random.seed(1024)
        data = np.random.random((10 * dp_group.nranks, 100)).reshape((dp_group.nranks, -1, 100))
        data_part = paddle.to_tensor(data[dp_group.rank])
        paddle.distributed.collective.all_reduce(data_part)
        data_reduced = data_part
        data_sumed = np.sum(data, axis=0)
        np.testing.assert_allclose(data_sumed, data_reduced.numpy(), rtol=1e-08, atol=1e-08)
if __name__ == '__main__':
    runtime_main(TestDygrapgHybridDP, 'dp')