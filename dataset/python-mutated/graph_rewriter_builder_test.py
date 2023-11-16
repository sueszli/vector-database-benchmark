"""Tests for graph_rewriter_builder."""
import mock
import tensorflow as tf
from object_detection.builders import graph_rewriter_builder
from object_detection.protos import graph_rewriter_pb2

class QuantizationBuilderTest(tf.test.TestCase):

    def testQuantizationBuilderSetsUpCorrectTrainArguments(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch.object(tf.contrib.quantize, 'experimental_create_training_graph') as mock_quant_fn:
            with mock.patch.object(tf.contrib.layers, 'summarize_collection') as mock_summarize_col:
                graph_rewriter_proto = graph_rewriter_pb2.GraphRewriter()
                graph_rewriter_proto.quantization.delay = 10
                graph_rewriter_proto.quantization.weight_bits = 8
                graph_rewriter_proto.quantization.activation_bits = 8
                graph_rewrite_fn = graph_rewriter_builder.build(graph_rewriter_proto, is_training=True)
                graph_rewrite_fn()
                (_, kwargs) = mock_quant_fn.call_args
                self.assertEqual(kwargs['input_graph'], tf.get_default_graph())
                self.assertEqual(kwargs['quant_delay'], 10)
                mock_summarize_col.assert_called_with('quant_vars')

    def testQuantizationBuilderSetsUpCorrectEvalArguments(self):
        if False:
            i = 10
            return i + 15
        with mock.patch.object(tf.contrib.quantize, 'experimental_create_eval_graph') as mock_quant_fn:
            with mock.patch.object(tf.contrib.layers, 'summarize_collection') as mock_summarize_col:
                graph_rewriter_proto = graph_rewriter_pb2.GraphRewriter()
                graph_rewriter_proto.quantization.delay = 10
                graph_rewrite_fn = graph_rewriter_builder.build(graph_rewriter_proto, is_training=False)
                graph_rewrite_fn()
                (_, kwargs) = mock_quant_fn.call_args
                self.assertEqual(kwargs['input_graph'], tf.get_default_graph())
                mock_summarize_col.assert_called_with('quant_vars')
if __name__ == '__main__':
    tf.test.main()