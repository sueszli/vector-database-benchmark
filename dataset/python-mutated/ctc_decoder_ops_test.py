"""Tests for tensorflow.ctc_ops.ctc_decoder_ops."""
import itertools
import numpy as np
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import ctc_ops
from tensorflow.python.platform import test

def grouper(iterable, n, fillvalue=None):
    if False:
        while True:
            i = 10
    'Collect data into fixed-length chunks or blocks.'
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def flatten(list_of_lists):
    if False:
        while True:
            i = 10
    'Flatten one level of nesting.'
    return itertools.chain.from_iterable(list_of_lists)

class CTCGreedyDecoderTest(test.TestCase):

    def _testCTCDecoder(self, decoder, inputs, seq_lens, log_prob_truth, decode_truth, expected_err_re=None, **decoder_args):
        if False:
            while True:
                i = 10
        inputs_t = [ops.convert_to_tensor(x) for x in inputs]
        inputs_t = array_ops_stack.stack(inputs_t)
        with self.cached_session(use_gpu=False) as sess:
            (decoded_list, log_probability) = decoder(inputs_t, sequence_length=seq_lens, **decoder_args)
            decoded_unwrapped = list(flatten([(st.indices, st.values, st.dense_shape) for st in decoded_list]))
            if expected_err_re is None:
                outputs = sess.run(decoded_unwrapped + [log_probability])
                output_sparse_tensors = list(grouper(outputs[:-1], 3))
                output_log_probability = outputs[-1]
                self.assertEqual(len(output_sparse_tensors), len(decode_truth))
                for (out_st, truth_st, tf_st) in zip(output_sparse_tensors, decode_truth, decoded_list):
                    self.assertAllEqual(out_st[0], truth_st[0])
                    self.assertAllEqual(out_st[1], truth_st[1])
                    self.assertAllEqual(out_st[2], truth_st[2])
                    self.assertEqual([None, truth_st[0].shape[1]], tf_st.indices.get_shape().as_list())
                    self.assertEqual([None], tf_st.values.get_shape().as_list())
                    self.assertShapeEqual(truth_st[2], tf_st.dense_shape)
                self.assertAllClose(output_log_probability, log_prob_truth, atol=1e-06)
            else:
                with self.assertRaisesOpError(expected_err_re):
                    sess.run(decoded_unwrapped + [log_probability])

    @test_util.run_deprecated_v1
    def testCTCGreedyDecoder(self):
        if False:
            i = 10
            return i + 15
        'Test two batch entries - best path decoder.'
        max_time_steps = 6
        seq_len_0 = 4
        input_prob_matrix_0 = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.4, 0.6], [0.0, 0.0, 0.4, 0.6], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        input_log_prob_matrix_0 = np.log(input_prob_matrix_0)
        seq_len_1 = 5
        input_prob_matrix_1 = np.asarray([[0.1, 0.9, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.1, 0.9], [0.0, 0.9, 0.1, 0.1], [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        input_log_prob_matrix_1 = np.log(input_prob_matrix_1)
        inputs = np.array([np.vstack([input_log_prob_matrix_0[t, :], input_log_prob_matrix_1[t, :]]) for t in range(max_time_steps)])
        seq_lens = np.array([seq_len_0, seq_len_1], dtype=np.int32)
        log_prob_truth = np.array([np.sum(-np.log([1.0, 0.6, 0.6, 0.9])), np.sum(-np.log([0.9, 0.9, 0.9, 0.9, 0.9]))], np.float32)[:, np.newaxis]
        decode_truth = [(np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], dtype=np.int64), np.array([0, 1, 1, 1, 0], dtype=np.int64), np.array([2, 3], dtype=np.int64))]
        self._testCTCDecoder(ctc_ops.ctc_greedy_decoder, inputs, seq_lens, log_prob_truth, decode_truth)
        blank_index = 2
        inputs = np.concatenate((inputs[:, :, :blank_index], inputs[:, :, -1:], inputs[:, :, blank_index:-1]), axis=2)
        self._testCTCDecoder(ctc_ops.ctc_greedy_decoder, inputs, seq_lens, log_prob_truth, decode_truth, blank_index=2)
        self._testCTCDecoder(ctc_ops.ctc_greedy_decoder, inputs, seq_lens, log_prob_truth, decode_truth, blank_index=-2)

    @test_util.run_deprecated_v1
    def testCTCDecoderBeamSearch(self):
        if False:
            print('Hello World!')
        'Test one batch, two beams - hibernating beam search.'
        depth = 6
        seq_len_0 = 5
        input_prob_matrix_0 = np.asarray([[0.30999, 0.309938, 0.0679938, 0.0673362, 0.0708352, 0.173908], [0.215136, 0.439699, 0.0370931, 0.0393967, 0.0381581, 0.230517], [0.199959, 0.489485, 0.0233221, 0.0251417, 0.0233289, 0.238763], [0.279611, 0.452966, 0.0204795, 0.0209126, 0.0194803, 0.20655], [0.51286, 0.288951, 0.0243026, 0.0220788, 0.0219297, 0.129878], [0.155251, 0.164444, 0.173517, 0.176138, 0.169979, 0.160671]], dtype=np.float32)
        input_prob_matrix_0 = input_prob_matrix_0 + 2.0
        inputs = [input_prob_matrix_0[t, :][np.newaxis, :] for t in range(seq_len_0)] + 2 * [np.zeros((1, depth), dtype=np.float32)]
        seq_lens = np.array([seq_len_0], dtype=np.int32)
        log_prob_truth = np.array([-5.811451, -6.63339], np.float32)[np.newaxis, :]
        decode_truth = [(np.array([[0, 0], [0, 1]], dtype=np.int64), np.array([1, 0], dtype=np.int64), np.array([1, 2], dtype=np.int64)), (np.array([[0, 0]], dtype=np.int64), np.array([1], dtype=np.int64), np.array([1, 1], dtype=np.int64))]
        self._testCTCDecoder(ctc_ops.ctc_beam_search_decoder, inputs, seq_lens, log_prob_truth, decode_truth, beam_width=2, top_paths=2)
        with self.assertRaisesRegex(errors.InvalidArgumentError, '.*requested more paths than the beam width.*'):
            self._testCTCDecoder(ctc_ops.ctc_beam_search_decoder, inputs, seq_lens, log_prob_truth, decode_truth, beam_width=2, top_paths=3)
if __name__ == '__main__':
    test.main()