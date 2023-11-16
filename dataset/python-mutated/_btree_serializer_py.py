"""B+Tree index parsing."""
from __future__ import absolute_import
from bzrlib import static_tuple

def _parse_leaf_lines(bytes, key_length, ref_list_length):
    if False:
        for i in range(10):
            print('nop')
    lines = bytes.split('\n')
    nodes = []
    as_st = static_tuple.StaticTuple.from_sequence
    stuple = static_tuple.StaticTuple
    for line in lines[1:]:
        if line == '':
            return nodes
        elements = line.split('\x00', key_length)
        key = as_st(elements[:key_length]).intern()
        line = elements[-1]
        (references, value) = line.rsplit('\x00', 1)
        if ref_list_length:
            ref_lists = []
            for ref_string in references.split('\t'):
                ref_list = as_st([as_st(ref.split('\x00')).intern() for ref in ref_string.split('\r') if ref])
                ref_lists.append(ref_list)
            ref_lists = as_st(ref_lists)
            node_value = stuple(value, ref_lists)
        else:
            node_value = stuple(value, stuple())
        nodes.append((key, node_value))
    return nodes

def _flatten_node(node, reference_lists):
    if False:
        return 10
    'Convert a node into the serialized form.\n\n    :param node: A tuple representing a node (key_tuple, value, references)\n    :param reference_lists: Does this index have reference lists?\n    :return: (string_key, flattened)\n        string_key  The serialized key for referencing this node\n        flattened   A string with the serialized form for the contents\n    '
    if reference_lists:
        flattened_references = ['\r'.join(['\x00'.join(reference) for reference in ref_list]) for ref_list in node[3]]
    else:
        flattened_references = []
    string_key = '\x00'.join(node[1])
    line = '%s\x00%s\x00%s\n' % (string_key, '\t'.join(flattened_references), node[2])
    return (string_key, line)