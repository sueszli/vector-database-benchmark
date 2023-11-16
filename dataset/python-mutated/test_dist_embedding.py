import unittest
from test_dist_pnorm import parallelizer
import paddle
from paddle.distributed.fleet import auto
paddle.enable_static()

def make_program_lookup_table_v1_mp_dp():
    if False:
        i = 10
        return i + 15
    main_program = paddle.base.Program()
    start_program = paddle.base.Program()
    block = main_program.global_block()
    with paddle.static.program_guard(main_program, start_program):
        src_ids = paddle.static.data(name='src_ids', shape=[12, 512, 1], dtype='int64')
        src_ids.stop_gradient = True
        emb_out = block.create_var(name='emb_out', dtype='float32')
        w = paddle.create_parameter(attr=paddle.base.ParamAttr(name='emb_weight'), shape=[64, 128], dtype='float32', is_bias=False)
        block.append_op(type='lookup_table', outputs={'Out': emb_out}, inputs={'Ids': src_ids, 'W': w}, attrs={'is_sparse': False, 'is_distributed': False, 'remote_prefetch': False, 'padding_idx': None})
        loss = paddle.mean(emb_out)
        auto.shard_tensor(src_ids, auto.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y']), ['x', None, None])
        emb_weight = block.vars['emb_weight']
        auto.shard_tensor(emb_weight, auto.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y']), ['y', None])
    return (main_program, start_program, loss)

class TestDistPNorm(unittest.TestCase):

    def test_lookup_table_v1_mp_dp(self):
        if False:
            print('Hello World!')
        for rank in range(4):
            (dist_main_prog, dist_context) = parallelizer(make_program_lookup_table_v1_mp_dp, rank)
            ops = dist_main_prog.global_block().ops
            op_types = []
            for op in ops:
                op_types.append(op.type)
            assert op_types == ['reshape2', 'c_embedding', 'c_allreduce_sum', 'reduce_mean', 'fill_constant', 'reduce_mean_grad', 'c_embedding_grad', 'c_allreduce_sum', 'scale']
if __name__ == '__main__':
    unittest.main()