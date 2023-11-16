"""Generates GraphDef test data for Merger.

Constructs chunked protos test data containing GraphDefs with lots of nodes and
large nodes for Merger::Read and Merger::Merge.

Example command:

bazel run tensorflow/tools/proto_splitter/testdata:split_graph_def_gen -- \\
  --path /tmp \\
  --graph_type=split-lots-nodes,split-large-nodes,split-large-constant \\
  --export=pb,cpb
"""
from collections.abc import Sequence
import os
from absl import app
from absl import flags
from absl import logging
from tensorflow.core.framework import graph_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.tools.proto_splitter import constants
from tensorflow.tools.proto_splitter import split_graph_def
from tensorflow.tools.proto_splitter.python import test_util
LOTS_NODES_SIZES = [95] * 15
LARGE_NODES_SIZES = [50, 95, 95, 95, 50, 95]
LARGE_CONSTANT_SIZES = [50, 50, 1000, 50, 1000]

def _split_and_write(path: str, graph_def: graph_pb2.GraphDef, max_size: int, export_files: Sequence[str]):
    if False:
        i = 10
        return i + 15
    'Writes the .pb, .pbtxt and .cpb files for a GraphDef.'
    constants.debug_set_max_size(max_size)
    if 'pbtxt' in export_files:
        output_path = f'{path}.pbtxt'
        file_io.write_string_to_file(output_path, str(graph_def))
        logging.info('  %s written', output_path)
    if 'pb' in export_files:
        output_path = f'{path}.pb'
        file_io.write_string_to_file(output_path, graph_def.SerializeToString())
        logging.info('  %s written', output_path)
    if 'cpb' in export_files:
        splitter = split_graph_def.GraphDefSplitter(graph_def)
        splitter.write(path)
        (chunks, _) = splitter.split()
        if len(chunks) > 1:
            logging.info('  %s.cpb written', path)
        else:
            raise RuntimeError('For some reason this graph was not chunked, so a .cpb file was not produced. Raising an error since this should not be the case.')

def split_lots_nodes(path: str, export_files: Sequence[str]):
    if False:
        return 10
    'GraphDef with lots of nodes.'
    graph_def = test_util.make_graph_def_with_constant_nodes(LOTS_NODES_SIZES)
    _split_and_write(path, graph_def, 500, export_files)

def split_large_nodes(path: str, export_files: Sequence[str]):
    if False:
        print('Hello World!')
    'GraphDef with large nodes.'
    graph_def = test_util.make_graph_def_with_constant_nodes(LARGE_NODES_SIZES)
    _split_and_write(path, graph_def, 200, export_files)

def split_large_constant(path: str, export_files: Sequence[str]):
    if False:
        i = 10
        return i + 15
    'GraphDef with large constant nodes.'
    graph_def = test_util.make_graph_def_with_constant_nodes(LARGE_CONSTANT_SIZES)
    _split_and_write(path, graph_def, 500, export_files)

def function_lots_of_nodes(path: str, export_files: Sequence[str]):
    if False:
        return 10
    'Generates a proto of GraphDef with a FunctionDef that have many nodes.'
    graph_def = test_util.make_graph_def_with_constant_nodes([], fn=LOTS_NODES_SIZES)
    _split_and_write(path, graph_def, 500, export_files)

def function_large_nodes(path: str, export_files: Sequence[str]):
    if False:
        for i in range(10):
            print('nop')
    graph_def = test_util.make_graph_def_with_constant_nodes([], fn=LARGE_NODES_SIZES)
    _split_and_write(path, graph_def, 200, export_files)

def graph_def_and_function(path: str, export_files: Sequence[str]):
    if False:
        while True:
            i = 10
    graph_def = test_util.make_graph_def_with_constant_nodes([50, 50, 50, 50, 50, 50], fn1=[50, 50, 50], fn2=[50], fn3=[50], fn4=[50])
    _split_and_write(path, graph_def, 200, export_files)
VALID_GRAPH_TYPES = {'split-lots-nodes': split_lots_nodes, 'split-large-nodes': split_large_nodes, 'split-large-constant': split_large_constant, 'function-lots-of-nodes': function_lots_of_nodes, 'function-large-nodes': function_large_nodes, 'graph-def-and-function': graph_def_and_function}
ALL_GRAPH_TYPES = ', '.join(VALID_GRAPH_TYPES.keys())
SPLITTER_TESTDATA_PATH = flags.DEFINE_string('path', None, help='Path to testdata directory.')
GRAPH_TYPES = flags.DEFINE_multi_string('graph_type', 'all', help=f'Type(s) of graph to export. Valid types: all, {ALL_GRAPH_TYPES}')
EXPORT_FILES = flags.DEFINE_multi_string('export', 'all', help='List of files to export. Valid options: all, pb, pbtxt, cpb')

def main(argv: Sequence[str]) -> None:
    if False:
        while True:
            i = 10
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    if 'all' in EXPORT_FILES.value:
        export_files = ['pb', 'pbtxt', 'cpb']
    else:
        export_files = EXPORT_FILES.value
    if 'all' in GRAPH_TYPES.value:
        graph_types = VALID_GRAPH_TYPES.keys()
    else:
        graph_types = GRAPH_TYPES.value
    for v in graph_types:
        if v not in VALID_GRAPH_TYPES:
            raise ValueError(f'Invalid flag passed to `graph_type`: {v}\nValid graph types: {ALL_GRAPH_TYPES}')
        logging.info('Generating graph %s', v)
        f = VALID_GRAPH_TYPES[v]
        f(os.path.join(SPLITTER_TESTDATA_PATH.value, v), export_files)
if __name__ == '__main__':
    app.run(main)