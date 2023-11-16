"""Imports a protobuf model as a graph in Tensorboard."""
import argparse
import sys
from absl import app
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.summary import summary
from tensorflow.python.tools import saved_model_utils
try:
    from tensorflow.contrib.tensorrt.ops.gen_trt_engine_op import *
except ImportError:
    pass

def import_to_tensorboard(model_dir, log_dir, tag_set):
    if False:
        return 10
    "View an SavedModel as a graph in Tensorboard.\n\n  Args:\n    model_dir: The directory containing the SavedModel to import.\n    log_dir: The location for the Tensorboard log to begin visualization from.\n    tag_set: Group of tag(s) of the MetaGraphDef to load, in string format,\n      separated by ','. For tag-set contains multiple tags, all tags must be\n      passed in.\n  Usage: Call this function with your SavedModel location and desired log\n    directory. Launch Tensorboard by pointing it to the log directory. View your\n    imported SavedModel as a graph.\n  "
    with session.Session(graph=ops.Graph()) as sess:
        input_graph_def = saved_model_utils.get_meta_graph_def(model_dir, tag_set).graph_def
        importer.import_graph_def(input_graph_def)
        pb_visual_writer = summary.FileWriter(log_dir)
        pb_visual_writer.add_graph(sess.graph)
        print('Model Imported. Visualize by running: tensorboard --logdir={}'.format(log_dir))

def main(_):
    if False:
        for i in range(10):
            print('nop')
    import_to_tensorboard(FLAGS.model_dir, FLAGS.log_dir, FLAGS.tag_set)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--model_dir', type=str, default='', required=True, help='The directory containing the SavedModel to import.')
    parser.add_argument('--log_dir', type=str, default='', required=True, help='The location for the Tensorboard log to begin visualization from.')
    parser.add_argument('--tag_set', type=str, default='serve', required=False, help="tag-set of graph in SavedModel to load, separated by ','")
    (FLAGS, unparsed) = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)