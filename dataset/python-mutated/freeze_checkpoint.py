import tensorflow as tf
from bigdl.dllib.utils.tf import export_tf
from optparse import OptionParser
from bigdl.dllib.utils.log4Error import *

def ckpt_to_frozen_graph(options):
    if False:
        return 10
    with tf.gfile.GFile(options.pbPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        var_list_name = [node.name + ':0' for node in graph_def.node if node.op in ['Variable', 'VariableV2', 'VarHandleOp']]
    with tf.Session() as sess:
        graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='')
        var_list = [graph.get_tensor_by_name(name) for name in var_list_name]
        for v in var_list:
            tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, v)
        saver = tf.train.Saver(var_list)
        saver.restore(sess, options.ckptPath)
        input_names = options.inputsName.split(',')
        output_names = options.outputsName.split(',')
        input_tensors = [graph.get_tensor_by_name(name) for name in input_names]
        output_tensors = [graph.get_tensor_by_name(name) for name in output_names]
        export_tf(sess, options.outputDir, inputs=input_tensors, outputs=output_tensors)
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--pbPath', dest='pbPath', help='The path to a TensorFlow pb file')
    parser.add_option('--ckptPath', dest='ckptPath', help='The path to a TensorFlow chekpoint file')
    parser.add_option('--inputsName', dest='inputsName', help='A comma separated list of Tensor names as the model inputs, e.g. input_0:0,input_1:0.')
    parser.add_option('--outputsName', dest='outputsName', help='A comma separated list of Tensor names as the model outputs, e.g. output_0:0,output_1:0.')
    parser.add_option('-o', '--outputDir', dest='outputDir', default='.')
    import sys
    (options, args) = parser.parse_args(sys.argv)
    invalidInputError(options.pbPath is not None, '--pbPath must be provided')
    invalidInputError(options.ckptPath is not None, '--ckptPath must be provided')
    invalidInputError(options.inputsName is not None, '--inputsName must be provided')
    invalidInputError(options.outputsName is not None, '--outputsName must be provided')
    ckpt_to_frozen_graph(options)