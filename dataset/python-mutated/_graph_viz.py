import os

def _shape_notation(int_shape):
    if False:
        for i in range(10):
            print('nop')
    X = ['S', 'B', 'C', 'H', 'W']
    return [X[i] for i in int_shape]

def plot_graph(graph, graph_img_path='graph.png', show_coreml_mapped_shapes=False):
    if False:
        while True:
            i = 10
    '\n    Plot graph using pydot\n\n    It works in two steps:\n    1. Add nodes to pydot\n    2. connect nodes added in pydot\n\n    :param graph\n    :return: writes down a png/pdf file using dot\n    '
    try:
        import pydot_ng as pydot
    except:
        try:
            import pydotplus as pydot
        except:
            try:
                import pydot
            except:
                return None
    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')
    graph_inputs = []
    for input_ in graph.inputs:
        if show_coreml_mapped_shapes:
            if input_[0] in graph.onnx_coreml_shape_mapping:
                shape = tuple(_shape_notation(graph.onnx_coreml_shape_mapping[input_[0]]))
            else:
                shape = 'NA, '
        else:
            shape = tuple(input_[2])
        label = '%s\n|{|%s}|{{%s}|{%s}}' % ('Input', input_[0], '', str(shape))
        pydot_node = pydot.Node(input_[0], label=label)
        dot.add_node(pydot_node)
        graph_inputs.append(input_[0])
    for node in graph.nodes:
        inputlabels = ''
        for input_ in node.inputs:
            if show_coreml_mapped_shapes:
                if input_ in graph.onnx_coreml_shape_mapping:
                    inputlabels += str(tuple(_shape_notation(graph.onnx_coreml_shape_mapping[input_]))) + ', '
                else:
                    inputlabels += 'NA, '
            elif input_ in graph.shape_dict:
                inputlabels += str(tuple(graph.shape_dict[input_])) + ', '
            else:
                inputlabels += 'NA, '
        outputlabels = ''
        for output_ in node.outputs:
            if show_coreml_mapped_shapes:
                if output_ in graph.onnx_coreml_shape_mapping:
                    outputlabels += str(tuple(_shape_notation(graph.onnx_coreml_shape_mapping[output_]))) + ', '
                else:
                    outputlabels += 'NA, '
            elif output_ in graph.shape_dict:
                outputlabels += str(tuple(graph.shape_dict[output_])) + ', '
            else:
                outputlabels += 'NA, '
        output_names = ', '.join([output_ for output_ in node.outputs])
        input_names = ', '.join([input_ for input_ in node.inputs])
        label = '%s\n|{{%s}|{%s}}|{{%s}|{%s}}' % (node.op_type, input_names, output_names, inputlabels, outputlabels)
        pydot_node = pydot.Node(node.name, label=label)
        dot.add_node(pydot_node)
    for node in graph.nodes:
        for child in node.children:
            dot.add_edge(pydot.Edge(node.name, child.name))
        for input_ in node.inputs:
            if input_ in graph_inputs:
                dot.add_edge(pydot.Edge(input_, node.name))
    (_, extension) = os.path.splitext(graph_img_path)
    if not extension:
        extension = 'pdf'
    else:
        extension = extension[1:]
    dot.write(graph_img_path, format=extension)