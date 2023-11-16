import onnx
from onnx import version_converter, helper, ModelProto

def add_value_info_for_constants(model: onnx.ModelProto):
    if False:
        print('Hello World!')
    "\n    Currently onnx.shape_inference doesn't use the shape of initializers, so add\n    that info explicitly as ValueInfoProtos.\n    Mutates the model.\n    Args:\n        model: The ModelProto to update.\n    "
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph: onnx.GraphProto):
        if False:
            while True:
                i = 10
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            if init.name in inputs:
                continue
            elem_type = init.data_type
            shape = init.dims
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField('shape'):
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim
            graph_input = graph.input.add()
            graph_input.name = vi.name
            graph_input.type.tensor_type.elem_type = elem_type
        for node in graph.node:
            for attr in node.attribute:
                if attr.ref_attr_name != '':
                    continue
                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)
    return add_const_value_infos_to_graph(model.graph)

def summarize_model(input: ModelProto):
    if False:
        while True:
            i = 10
    return f'Inputs {len(input.graph.input)} Nodes {len(input.graph.node)} Initializer {len(input.graph.initializer)} Value info {len(input.graph.value_info)}'
model = onnx.load('C:\\Users\\agibs\\Downloads\\V9\\V9\\best_bracket.onnx')
kotlin_model = onnx.load('C:\\Users\\agibs\\Documents\\GitHub\\dl4j-PR-split\\deeplearning4j\\nd4j\\samediff-import\\samediff-import-onnx\\input-adjusted-model.onnx')
input_names_2 = [node.name for node in kotlin_model.graph.node]
input_init__names_2 = [initializer.name for initializer in kotlin_model.graph.initializer]
add_value_info_for_constants(model)
input_names = [node.name for node in model.graph.node]
input_init__names = [initializer.name for initializer in model.graph.initializer]
input_val_info__names = [value_info.name for value_info in model.graph.value_info]
converted_model = version_converter.convert_version(kotlin_model, 13)
converted_input_val_info__names = [value_info.name for value_info in converted_model.graph.value_info]
converted_node_names = [node.name for node in converted_model.graph.node]
onnx.save(converted_model, 'output.onnx')
print('Converted model')