import argparse
import os
import sys
import unittest
import paddle
from paddle.base.framework import IrGraph
from paddle.framework import core
paddle.enable_static()

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='', help='A path to a model.')
    parser.add_argument('--save_graph_dir', type=str, default='', help='A path to save the graph.')
    parser.add_argument('--save_graph_name', type=str, default='', help='A name to save the graph. Default - name from model path will be used')
    (test_args, args) = parser.parse_known_args(namespace=unittest)
    return (test_args, sys.argv[:1] + args)

def generate_dot_for_model(model_path, save_graph_dir, save_graph_name):
    if False:
        i = 10
        return i + 15
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    inference_scope = paddle.static.global_scope()
    with paddle.static.scope_guard(inference_scope):
        if os.path.exists(os.path.join(model_path, '__model__')):
            [inference_program, feed_target_names, fetch_targets] = paddle.static.io.load_inference_model(model_path, exe, model_filename='__model__')
        else:
            [inference_program, feed_target_names, fetch_targets] = paddle.static.load_inference_model(model_path, exe, model_filename='model', params_filename='params')
        graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
        if not os.path.exists(save_graph_dir):
            os.makedirs(save_graph_dir)
        model_name = os.path.basename(os.path.normpath(save_graph_dir))
        if save_graph_name == '':
            save_graph_name = model_name
        graph.draw(save_graph_dir, save_graph_name, graph.all_op_nodes())
        print('Success! Generated dot and pdf files for {} model, that can be found at {} named {}.\n'.format(model_name, save_graph_dir, save_graph_name))
if __name__ == '__main__':
    global test_args
    (test_args, remaining_args) = parse_args()
    generate_dot_for_model(test_args.model_path, test_args.save_graph_dir, test_args.save_graph_name)