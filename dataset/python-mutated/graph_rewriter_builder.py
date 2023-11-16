"""Functions for quantized training and evaluation."""
import tensorflow as tf

def build(graph_rewriter_config, is_training):
    if False:
        while True:
            i = 10
    'Returns a function that modifies default graph based on options.\n\n  Args:\n    graph_rewriter_config: graph_rewriter_pb2.GraphRewriter proto.\n    is_training: whether in training of eval mode.\n  '

    def graph_rewrite_fn():
        if False:
            print('Hello World!')
        'Function to quantize weights and activation of the default graph.'
        if graph_rewriter_config.quantization.weight_bits != 8 or graph_rewriter_config.quantization.activation_bits != 8:
            raise ValueError('Only 8bit quantization is supported')
        if is_training:
            tf.contrib.quantize.experimental_create_training_graph(input_graph=tf.get_default_graph(), quant_delay=graph_rewriter_config.quantization.delay)
        else:
            tf.contrib.quantize.experimental_create_eval_graph(input_graph=tf.get_default_graph())
        tf.contrib.layers.summarize_collection('quant_vars')
    return graph_rewrite_fn