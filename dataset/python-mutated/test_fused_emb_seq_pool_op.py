import platform
import unittest
import numpy as np
from op_test import OpTest, paddle_static_guard, skip_check_grad_ci
import paddle
import paddle.version as ver
from paddle.incubate.layers.nn import fused_embedding_seq_pool

@skip_check_grad_ci(reason="check_grad is called when ver.mkl() == ONand 'Linux' in platform.platform().")
class TestFusedEmbeddingSeqPoolOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'fused_embedding_seq_pool'
        self.emb_size = 6
        self.table = np.random.random((17, self.emb_size)).astype('float64')
        self.ids = np.array([[[4], [3]], [[4], [3]], [[2], [1]], [[16], [1]]]).astype('int64')
        ids_expand = np.expand_dims(self.ids, axis=1)
        self.lod = [[3, 1]]
        self.attrs = {'is_sparse': True}
        self.inputs = {'W': self.table, 'Ids': (ids_expand, self.lod)}
        self.outputs = {'Out': np.reshape(np.array([self.table[[4, 3]] + self.table[[4, 3]] + self.table[[2, 1]], self.table[[16, 1]]]), [len(self.lod[0]), 2 * self.emb_size])}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        if ver.mkl() == 'ON' and 'Linux' in platform.platform():
            self.attrs = {'is_sparse': False}
            self.check_grad(['W'], 'Out', no_grad_set=['Ids'], check_dygraph=False)

class TestLookupTableOpWithPadding(TestFusedEmbeddingSeqPoolOp):

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        if ver.mkl() == 'ON' and 'Linux' in platform.platform():
            ids = np.squeeze(self.ids, axis=2)
            padding_idx = np.random.choice(ids.flatten(), 1)[0]
            output = []
            index = 0
            for count in self.lod[0]:
                arr = ids[index:count + index]
                out = np.reshape(self.table[arr.flatten()], [arr.shape[0], arr.shape[1], self.emb_size])
                idx = np.argwhere(arr == padding_idx)
                for item in idx:
                    out[item[0], item[1], :] = np.zeros(self.emb_size)
                output.append(np.sum(out, 0))
                index += count
            self.outputs = {'Out': np.reshape(np.array(output), [len(self.lod[0]), 2 * self.emb_size])}
            self.attrs = {'padding_idx': int(padding_idx)}
            self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            return 10
        if ver.mkl() == 'ON' and 'Linux' in platform.platform():
            ids = np.squeeze(self.ids, axis=2)
            padding_idx = np.random.choice(ids.flatten(), 1)[0]
            self.attrs = {'padding_idx': int(padding_idx), 'is_sparse': False}
            self.check_grad(['W'], 'Out', no_grad_set=['Ids'], check_dygraph=False)

class TestFusedEmbeddingSeqPoolApi(unittest.TestCase):

    def test_api(self):
        if False:
            while True:
                i = 10
        with paddle_static_guard():
            if ver.mkl() == 'ON' and 'Linux' in platform.platform():
                from paddle import base
                dict_size = 20
                data_t = paddle.static.data(name='word', shape=[-1, 1], dtype='int64', lod_level=1)
                padding_idx = np.random.randint(1, 10)
                out = fused_embedding_seq_pool(input=data_t, size=[dict_size, 32], param_attr='w', padding_idx=padding_idx, is_sparse=False)
                place = base.CPUPlace()
                exe = base.Executor(place)
                exe.run(base.default_startup_program())
                x_tensor = base.core.LoDTensor()
                idxs = np.random.randint(1, 10, 8).astype('int64')
                x_tensor.set(idxs, place)
                x_tensor.set_recursive_sequence_lengths([[4, 4]])
                ret = exe.run(feed={'word': x_tensor}, fetch_list=[out])
if __name__ == '__main__':
    unittest.main()