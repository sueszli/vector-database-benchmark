"""A tool that finds all subgraphs of a given size in a TF graph.

The subgraph patterns are sorted by occurrence, and only the transitive fanin
part of the graph with regard to the fetch nodes is considered.
"""
import argparse
import sys
from absl import app
from tensorflow.python.grappler import _pywrap_graph_analyzer as tf_wrap

def main(_):
    if False:
        for i in range(10):
            print('nop')
    tf_wrap.GraphAnalyzer(FLAGS.input, FLAGS.n)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='Input file path for a TensorFlow MetaGraphDef.')
    parser.add_argument('--n', type=int, default=None, help='The size of the subgraphs.')
    (FLAGS, unparsed) = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)