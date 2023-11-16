from onnx.defs import get_all_schemas
from onnx import NodeProto, GraphProto
from google.protobuf import text_format
import onnx.helper
nodes = []
schemas = get_all_schemas()

def load_node(input_str):
    if False:
        print('Hello World!')
    '\n    Return a node\n    :param input_str:\n    :return:\n    '
    node_proto = NodeProto()
    text_format.Parse(input_str, node_proto)
    return node_proto

def convert_attr_type_to_enum(attr_value):
    if False:
        return 10
    '\n    Pass in an attribute from OpDescriptor and\n    get back out the equivalent enum value\n    for conversion to an attribute proto.\n    :param attr_value:  the attribute value\n    :return:\n    '
    if str(attr_value.type) == 'AttrType.INTS':
        return 7
    elif str(attr_value.type) == 'AttrType.UNDEFINED':
        return 0
    elif str(attr_value.type) == 'AttrType.FLOATS':
        return 6
    elif str(attr_value.type) == 'AttrType.GRAPH':
        return 5
    elif str(attr_value.type) == 'AttrType.GRAPHS':
        return 10
    elif str(attr_value.type) == 'AttrType.INT':
        return 2
    elif str(attr_value.type) == 'AttrType.STRING':
        return 3
    elif str(attr_value.type) == 'AttrType.TENSOR':
        return 4
    elif str(attr_value.type) == 'AttrType.TENSORS':
        return 9
    elif str(attr_value.type) == 'AttrType.SPARSE_TENSOR':
        return 11
    elif str(attr_value.type) == 'AttrType.SPARSE_TENSORS':
        return 12
    elif str(attr_value.type) == 'AttrType.FLOAT':
        return 1
    elif str(attr_value.type) == 'AttrType.STRINGS':
        return 8
    else:
        raise Exception('Invalid type passed in')

def create_node_from_schema(schema):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert an OpSchema to a NodeProto\n    :param schema:  the input OpSchema\n    :return: the equivalent NodeProto\n    '
    node_proto = NodeProto()
    for attribute in schema.attributes:
        attr_value = schema.attributes[attribute]
        if attr_value.default_value.name == '':
            attr_value_new = onnx.helper.make_attribute(attr_value.name, '')
            attr_value_new.type = convert_attr_type_to_enum(attr_value)
            node_proto.attribute.append(attr_value_new)
        else:
            node_proto.attribute.append(attr_value.default_value)
    node_proto.op_type = schema.name
    node_proto.doc_string = schema.doc
    node_proto.name = schema.name
    for input_arr in schema.inputs:
        input_types = input_arr.types
        type_attr = onnx.helper.make_attribute(input_arr.name + '-types', [str(data_type).replace('tensor(', '').replace(')', '') for data_type in input_types])
        node_proto.attribute.append(type_attr)
        if node_proto.input is None:
            node_proto.input = []
        node_proto.input.append(input_arr.name)
    for output_arr in schema.outputs:
        if node_proto.output is None:
            node_proto.output = []
            output_types = output_arr.types
            type_attr = onnx.helper.make_attribute(output_arr.name + '-types', [str(data_type).replace('tensor(', '').replace(')', '') for data_type in output_types])
            node_proto.attribute.append(type_attr)
        node_proto.output.append(output_arr.name)
    return node_proto
nodes = [create_node_from_schema(schema) for schema in sorted(schemas, key=lambda s: s.name)]
graph_proto = GraphProto()
graph_proto.node.extend(nodes)
text_proto = text_format.MessageToString(graph_proto)
with open('onnx-op-defs.pb', 'wb') as f:
    f.write(graph_proto.SerializeToString())
with open('onnx-op-def.pbtxt', 'w+') as f:
    f.write(text_proto)