import os
import unittest
import numpy as np
import transformer_model
from feed_data_reader import FeedDataReader
from parallel_executor_test_base import DeviceType, TestParallelExecutorBase
import paddle
from paddle.base import core
from paddle.dataset import wmt16
os.environ['CPU_NUM'] = str(4)

class ModelHyperParams:
    src_vocab_size = 10000
    src_pad_idx = src_vocab_size
    trg_vocab_size = 10000
    trg_pad_idx = trg_vocab_size
    pos_pad_idx = 0
    max_length = 50
    d_model = 512
    d_inner_hid = 1024
    d_key = 64
    d_value = 64
    n_head = 8
    n_layer = 4
    dropout = 0.1

def prepare_batch_input(insts, src_pad_idx, trg_pad_idx, n_head):
    if False:
        print('Hello World!')
    '\n    Pad the instances to the max sequence length in batch, and generate the\n    corresponding position data and attention bias. Then, convert the numpy\n    data to tensors and return a dict mapping names to tensors.\n    '

    def __pad_batch_data(insts, pad_idx, is_target=False, return_pos=True, return_attn_bias=True, return_max_len=True):
        if False:
            while True:
                i = 10
        '\n        Pad the instances to the max sequence length in batch, and generate the\n        corresponding position data and attention bias.\n        '
        return_list = []
        max_len = max((len(inst) for inst in insts))
        inst_data = np.array([inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_data.astype('int64').reshape([-1, 1])]
        if return_pos:
            inst_pos = np.array([[pos_i + 1 if w_i != pad_idx else 0 for (pos_i, w_i) in enumerate(inst)] for inst in inst_data])
            return_list += [inst_pos.astype('int64').reshape([-1, 1])]
        if return_attn_bias:
            if is_target:
                slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
                slf_attn_bias_data = np.triu(slf_attn_bias_data, 1).reshape([-1, 1, max_len, max_len])
                slf_attn_bias_data = np.tile(slf_attn_bias_data, [1, n_head, 1, 1]) * [-1000000000.0]
            else:
                slf_attn_bias_data = np.array([[0] * len(inst) + [-1000000000.0] * (max_len - len(inst)) for inst in insts])
                slf_attn_bias_data = np.tile(slf_attn_bias_data.reshape([-1, 1, 1, max_len]), [1, n_head, max_len, 1])
            return_list += [slf_attn_bias_data.astype('float32')]
        if return_max_len:
            return_list += [max_len]
        return return_list if len(return_list) > 1 else return_list[0]
    (src_word, src_pos, src_slf_attn_bias, src_max_len) = __pad_batch_data([inst[0] for inst in insts], src_pad_idx, is_target=False)
    (trg_word, trg_pos, trg_slf_attn_bias, trg_max_len) = __pad_batch_data([inst[1] for inst in insts], trg_pad_idx, is_target=True)
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :], [1, 1, trg_max_len, 1]).astype('float32')
    lbl_word = __pad_batch_data([inst[2] for inst in insts], trg_pad_idx, False, False, False, False)
    lbl_weight = (lbl_word != trg_pad_idx).astype('float32').reshape([-1, 1])
    return [src_word, src_pos, trg_word, trg_pos, src_slf_attn_bias, trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight]
feed_data_reader = None

def transformer(use_feed):
    if False:
        while True:
            i = 10
    assert not use_feed, "transfomer doesn't support feed yet"
    return transformer_model.transformer(ModelHyperParams.src_vocab_size + 1, ModelHyperParams.trg_vocab_size + 1, ModelHyperParams.max_length + 1, ModelHyperParams.n_layer, ModelHyperParams.n_head, ModelHyperParams.d_key, ModelHyperParams.d_value, ModelHyperParams.d_model, ModelHyperParams.d_inner_hid, ModelHyperParams.dropout, ModelHyperParams.src_pad_idx, ModelHyperParams.trg_pad_idx, ModelHyperParams.pos_pad_idx)

def get_feed_data_reader():
    if False:
        while True:
            i = 10
    global feed_data_reader
    if feed_data_reader is not None:
        return feed_data_reader
    reader = paddle.batch(wmt16.train(ModelHyperParams.src_vocab_size, ModelHyperParams.trg_vocab_size), batch_size=transformer_model.batch_size)
    all_batch_tensors = []
    for batch in reader():
        tensors = []
        for tensor in prepare_batch_input(batch, ModelHyperParams.src_pad_idx, ModelHyperParams.trg_pad_idx, ModelHyperParams.n_head):
            tensors.append(np.array(tensor))
        all_batch_tensors.append(tensors)

    def __reader__():
        if False:
            print('Hello World!')
        yield from all_batch_tensors
    feed_data_reader = FeedDataReader(feed_list=transformer_model.build_inputs(ModelHyperParams.max_length + 1, ModelHyperParams.n_head), reader=__reader__)
    return feed_data_reader

class TestTransformer(TestParallelExecutorBase):

    def test_main(self):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_cuda():
            self.check_network_convergence(transformer, use_device=DeviceType.CUDA, feed_data_reader=get_feed_data_reader())
            self.check_network_convergence(transformer, use_device=DeviceType.CUDA, enable_sequential_execution=True, feed_data_reader=get_feed_data_reader())
        self.check_network_convergence(transformer, use_device=DeviceType.CPU, iter=2, feed_data_reader=get_feed_data_reader())
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()