"""Tests for GraphDef splitter."""
import itertools
from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import test
from tensorflow.tools.proto_splitter import chunk_pb2
from tensorflow.tools.proto_splitter import constants
from tensorflow.tools.proto_splitter import split_graph_def
from tensorflow.tools.proto_splitter import util
from tensorflow.tools.proto_splitter.python import test_util

class GraphDefSplitterTest(test.TestCase):

    def _make_graph_def_with_constant_nodes(self, node_sizes, dtype=None, **function_node_sizes):
        if False:
            for i in range(10):
                print('nop')
        return test_util.make_graph_def_with_constant_nodes(node_sizes, dtype, **function_node_sizes)

    def _copy_graph(self, graph_def):
        if False:
            i = 10
            return i + 15
        'Create a copy of GraphDef.'
        graph_def_copy = graph_pb2.GraphDef()
        graph_def_copy.CopyFrom(graph_def)
        return graph_def_copy

    def _assert_chunk_sizes(self, chunks, max_size):
        if False:
            print('Hello World!')
        'Asserts that all chunk proto sizes are <= max_size.'
        for chunk in chunks:
            if isinstance(chunk, message.Message):
                self.assertLessEqual(chunk.ByteSize(), max_size)

    def _assert_field_tags(self, expected_fields, actual_fields):
        if False:
            return 10
        self.assertLen(actual_fields, len(expected_fields))
        for (expected, actual) in zip(expected_fields, actual_fields):
            self.assertProtoEquals(expected, actual)

    def testSplitNoChunks(self):
        if False:
            print('Hello World!')
        sizes = [50, 100, 50, 50, 100]
        max_size = 500
        constants.debug_set_max_size(max_size)
        graph_def = self._make_graph_def_with_constant_nodes(sizes)
        s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
        (chunks, _) = s.split()
        self.assertLen(chunks, 1)
        self.assertProtoEquals(graph_def, chunks[0])

    def testLargeConstant(self):
        if False:
            while True:
                i = 10
        sizes = [50, 50, 1000, 50, 1000]
        max_size = 500
        constants.debug_set_max_size(max_size)
        graph_def = self._make_graph_def_with_constant_nodes(sizes)
        s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
        (chunks, chunked_message) = s.split()
        self.assertLen(chunks, 3)
        self._assert_chunk_sizes(chunks, max_size)
        self.assertEqual(graph_def.node[4].attr['value'].tensor.tensor_content, chunks[2])
        self.assertEqual(graph_def.node[2].attr['value'].tensor.tensor_content, chunks[1])
        self.assertLen(chunked_message.chunked_fields, 2)
        self.assertEqual(1, chunked_message.chunked_fields[0].message.chunk_index)
        self.assertEqual(2, chunked_message.chunked_fields[1].message.chunk_index)
        self._assert_field_tags(util.get_field_tag(graph_def, ['node', 2, 'attr', 'value', 'tensor', 'tensor_content']), chunked_message.chunked_fields[0].field_tag)
        self._assert_field_tags(util.get_field_tag(graph_def, ['node', 4, 'attr', 'value', 'tensor', 'tensor_content']), chunked_message.chunked_fields[1].field_tag)

    def testLotsOfNodes(self):
        if False:
            print('Hello World!')
        sizes = [95] * 15
        max_size = 500
        constants.debug_set_max_size(500)
        graph_def = self._make_graph_def_with_constant_nodes(sizes)
        s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
        (chunks, chunked_message) = s.split()
        self.assertLen(chunks, 3)
        self._assert_chunk_sizes(chunks, max_size)
        for (node, chunk) in zip(graph_def.node, itertools.chain(chunks[0].node, chunks[1].node, chunks[2].node)):
            self.assertProtoEquals(node, chunk)
        self.assertLen(chunked_message.chunked_fields, 2)
        self.assertEqual(1, chunked_message.chunked_fields[0].message.chunk_index)
        self.assertEqual(2, chunked_message.chunked_fields[1].message.chunk_index)
        self.assertEmpty(chunked_message.chunked_fields[0].field_tag)
        self.assertEmpty(chunked_message.chunked_fields[1].field_tag)

    def testLargeNodes(self):
        if False:
            i = 10
            return i + 15
        sizes = [50, 95, 95, 95, 50, 95]
        max_size = 200
        constants.debug_set_max_size(max_size)
        graph_def = self._make_graph_def_with_constant_nodes(sizes)
        s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
        (chunks, chunked_message) = s.split()
        self.assertLen(chunks, 5)
        self._assert_chunk_sizes(chunks, max_size)
        self.assertProtoEquals(graph_def.node[1], chunks[1])
        self.assertProtoEquals(graph_def.node[2], chunks[2])
        self.assertProtoEquals(graph_def.node[3], chunks[3])
        self.assertProtoEquals(graph_def.node[5], chunks[4])
        self.assertProtoEquals(graph_def.node[0], chunks[0].node[0])
        self.assertProtoEquals(graph_def.node[4], chunks[0].node[4])
        self.assertEqual(0, chunks[0].node[1].ByteSize())
        self.assertEqual(0, chunks[0].node[2].ByteSize())
        self.assertEqual(0, chunks[0].node[3].ByteSize())
        self.assertEqual(0, chunks[0].node[5].ByteSize())
        self.assertLen(chunked_message.chunked_fields, 4)
        self._assert_field_tags(util.get_field_tag(graph_def, ['node', 1]), chunked_message.chunked_fields[0].field_tag)
        self._assert_field_tags(util.get_field_tag(graph_def, ['node', 2]), chunked_message.chunked_fields[1].field_tag)
        self._assert_field_tags(util.get_field_tag(graph_def, ['node', 3]), chunked_message.chunked_fields[2].field_tag)
        self._assert_field_tags(util.get_field_tag(graph_def, ['node', 5]), chunked_message.chunked_fields[3].field_tag)

    def testFunctionLotsOfNodes(self):
        if False:
            for i in range(10):
                print('nop')
        sizes = []
        fn1 = [50, 50, 50, 50, 50]
        max_size = 200
        constants.debug_set_max_size(max_size)
        graph_def = self._make_graph_def_with_constant_nodes(sizes, fn=fn1)
        s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
        (chunks, chunked_message) = s.split()
        self.assertLen(chunks, 2)
        self.assertIsInstance(chunks[0], graph_pb2.GraphDef)
        self.assertIsInstance(chunks[1], function_pb2.FunctionDef)
        self._assert_chunk_sizes(chunks, max_size)
        for (node, chunk) in zip(graph_def.library.function[0].node_def, itertools.chain(chunks[0].library.function[0].node_def, chunks[1].node_def)):
            self.assertProtoEquals(node, chunk)
        expected_message = chunk_pb2.ChunkedMessage()
        text_format.Parse('\n        chunk_index: 0\n        chunked_fields {\n            field_tag {\n                field: 2\n            }\n            field_tag {\n                field: 1\n            }\n            field_tag {\n                index: 0\n            }\n            message {\n                chunk_index: 1\n            }\n        }', expected_message)
        self.assertProtoEquals(expected_message, chunked_message)

    def testFunctionLargeNodes(self):
        if False:
            while True:
                i = 10
        sizes = []
        fn1 = [500, 500, 50, 500]
        max_size = 200
        constants.debug_set_max_size(max_size)
        graph_def = self._make_graph_def_with_constant_nodes(sizes, fn=fn1)
        s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
        (chunks, _) = s.split()
        self.assertLen(chunks, 4)
        self._assert_chunk_sizes(chunks, max_size)
        self.assertIsInstance(chunks[0], graph_pb2.GraphDef)

        def get_const_value(index):
            if False:
                return 10
            node_def = graph_def.library.function[0].node_def[index]
            return node_def.attr['value'].tensor.tensor_content
        expected_values = [get_const_value(0), get_const_value(1), get_const_value(3)]
        for (expected, chunk) in zip(expected_values, chunks[1:]):
            self.assertEqual(expected, chunk)

    def testChunkGraphDefAndFunctions(self):
        if False:
            i = 10
            return i + 15
        sizes = [50, 50, 50, 50, 50, 50]
        fn1 = [50, 50, 50]
        fn2 = [50]
        fn3 = [50]
        fn4 = [50]
        max_size = 200
        constants.debug_set_max_size(max_size)
        graph_def = self._make_graph_def_with_constant_nodes(sizes, fn1=fn1, fn2=fn2, fn3=fn3, fn4=fn4)
        s = split_graph_def.GraphDefSplitter(self._copy_graph(graph_def))
        (chunks, _) = s.split()
        self.assertLen(chunks, 4)
        self._assert_chunk_sizes(chunks, max_size)
if __name__ == '__main__':
    test.main()