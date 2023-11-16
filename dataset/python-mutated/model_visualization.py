"""Utilities related to model visualization."""
import os
import sys
from keras.api_export import keras_export
from keras.utils import io_utils
try:
    import pydot_ng as pydot
except ImportError:
    try:
        import pydotplus as pydot
    except ImportError:
        try:
            import pydot
        except ImportError:
            pydot = None

def check_pydot():
    if False:
        while True:
            i = 10
    'Returns True if PyDot is available.'
    return pydot is not None

def check_graphviz():
    if False:
        return 10
    'Returns True if both PyDot and Graphviz are available.'
    if not check_pydot():
        return False
    try:
        pydot.Dot.create(pydot.Dot())
        return True
    except (OSError, pydot.InvocationException):
        return False

def add_edge(dot, src, dst):
    if False:
        print('Hello World!')
    if not dot.get_edge(src, dst):
        edge = pydot.Edge(src, dst)
        edge.set('penwidth', '2')
        dot.add_edge(edge)

def get_layer_activation_name(layer):
    if False:
        print('Hello World!')
    if hasattr(layer.activation, 'name'):
        activation_name = layer.activation.name
    elif hasattr(layer.activation, '__name__'):
        activation_name = layer.activation.__name__
    else:
        activation_name = str(layer.activation)
    return activation_name

def make_layer_label(layer, **kwargs):
    if False:
        print('Hello World!')
    class_name = layer.__class__.__name__
    show_layer_names = kwargs.pop('show_layer_names')
    show_layer_activations = kwargs.pop('show_layer_activations')
    show_dtype = kwargs.pop('show_dtype')
    show_shapes = kwargs.pop('show_shapes')
    show_trainable = kwargs.pop('show_trainable')
    if kwargs:
        raise ValueError(f'Invalid kwargs: {kwargs}')
    table = '<<table border="0" cellborder="1" bgcolor="black" cellpadding="10">'
    colspan = max(1, sum((int(x) for x in (show_dtype, show_shapes, show_trainable))))
    if show_layer_names:
        table += f'<tr><td colspan="{colspan}" bgcolor="black"><font point-size="16" color="white"><b>{layer.name}</b> ({class_name})</font></td></tr>'
    else:
        table += f'<tr><td colspan="{colspan}" bgcolor="black"><font point-size="16" color="white"><b>{class_name}</b></font></td></tr>'
    if show_layer_activations and hasattr(layer, 'activation') and (layer.activation is not None):
        table += f'<tr><td bgcolor="white" colspan="{colspan}"><font point-size="14">Activation: <b>{get_layer_activation_name(layer)}</b></font></td></tr>'
    cols = []
    if show_shapes:
        shape = None
        try:
            shape = layer.output.shape
        except ValueError:
            pass
        cols.append(f"""<td bgcolor="white"><font point-size="14">Output shape: <b>{shape or '?'}</b></font></td>""")
    if show_dtype:
        dtype = None
        try:
            dtype = layer.output.dtype
        except ValueError:
            pass
        cols.append(f"""<td bgcolor="white"><font point-size="14">Output dtype: <b>{dtype or '?'}</b></font></td>""")
    if show_trainable and hasattr(layer, 'trainable') and layer.weights:
        if layer.trainable:
            cols.append('<td bgcolor="forestgreen"><font point-size="14" color="white"><b>Trainable</b></font></td>')
        else:
            cols.append('<td bgcolor="firebrick"><font point-size="14" color="white"><b>Non-trainable</b></font></td>')
    if cols:
        colspan = len(cols)
    else:
        colspan = 1
    if cols:
        table += '<tr>' + ''.join(cols) + '</tr>'
    table += '</table>>'
    return table

def make_node(layer, **kwargs):
    if False:
        print('Hello World!')
    node = pydot.Node(str(id(layer)), label=make_layer_label(layer, **kwargs))
    node.set('fontname', 'Helvetica')
    node.set('border', '0')
    node.set('margin', '0')
    return node

@keras_export('keras.utils.model_to_dot')
def model_to_dot(model, show_shapes=False, show_dtype=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=200, subgraph=False, show_layer_activations=False, show_trainable=False, **kwargs):
    if False:
        i = 10
        return i + 15
    'Convert a Keras model to dot format.\n\n    Args:\n        model: A Keras model instance.\n        show_shapes: whether to display shape information.\n        show_dtype: whether to display layer dtypes.\n        show_layer_names: whether to display layer names.\n        rankdir: `rankdir` argument passed to PyDot,\n            a string specifying the format of the plot: `"TB"`\n            creates a vertical plot; `"LR"` creates a horizontal plot.\n        expand_nested: whether to expand nested Functional models\n            into clusters.\n        dpi: Image resolution in dots per inch.\n        subgraph: whether to return a `pydot.Cluster` instance.\n        show_layer_activations: Display layer activations (only for layers that\n            have an `activation` property).\n        show_trainable: whether to display if a layer is trainable.\n\n    Returns:\n        A `pydot.Dot` instance representing the Keras model or\n        a `pydot.Cluster` instance representing nested model if\n        `subgraph=True`.\n    '
    from keras.ops.function import make_node_key
    if not model.built:
        raise ValueError('This model has not yet been built. Build the model first by calling `build()` or by calling the model on a batch of data.')
    from keras.models import functional
    from keras.models import sequential
    if not check_pydot():
        raise ImportError('You must install pydot (`pip install pydot`) for model_to_dot to work.')
    if subgraph:
        dot = pydot.Cluster(style='dashed', graph_name=model.name)
        dot.set('label', model.name)
        dot.set('labeljust', 'l')
    else:
        dot = pydot.Dot()
        dot.set('rankdir', rankdir)
        dot.set('concentrate', True)
        dot.set('dpi', dpi)
        dot.set('splines', 'ortho')
        dot.set_node_defaults(shape='record')
    if kwargs.pop('layer_range', None) is not None:
        raise ValueError('Argument `layer_range` is no longer supported.')
    if kwargs:
        raise ValueError(f'Unrecognized keyword arguments: {kwargs}')
    kwargs = {'show_layer_names': show_layer_names, 'show_layer_activations': show_layer_activations, 'show_dtype': show_dtype, 'show_shapes': show_shapes, 'show_trainable': show_trainable}
    if isinstance(model, sequential.Sequential):
        layers = model.layers
    elif not isinstance(model, functional.Functional):
        node = make_node(model, **kwargs)
        dot.add_node(node)
        return dot
    else:
        layers = model._operations
    sub_n_first_node = {}
    sub_n_last_node = {}
    for (i, layer) in enumerate(layers):
        if expand_nested and isinstance(layer, functional.Functional):
            submodel = model_to_dot(layer, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, subgraph=True, show_layer_activations=show_layer_activations, show_trainable=show_trainable)
            sub_n_nodes = submodel.get_nodes()
            sub_n_first_node[layer.name] = sub_n_nodes[0]
            sub_n_last_node[layer.name] = sub_n_nodes[-1]
            dot.add_subgraph(submodel)
        else:
            node = make_node(layer, **kwargs)
            dot.add_node(node)
    if isinstance(model, sequential.Sequential):
        for i in range(len(layers) - 1):
            inbound_layer_id = str(id(layers[i]))
            layer_id = str(id(layers[i + 1]))
            add_edge(dot, inbound_layer_id, layer_id)
        return dot
    for (i, layer) in enumerate(layers):
        layer_id = str(id(layer))
        for (i, node) in enumerate(layer._inbound_nodes):
            node_key = make_node_key(layer, i)
            if node_key in model._nodes:
                for parent_node in node.parent_nodes:
                    inbound_layer = parent_node.operation
                    inbound_layer_id = str(id(inbound_layer))
                    if not expand_nested:
                        assert dot.get_node(inbound_layer_id)
                        assert dot.get_node(layer_id)
                        add_edge(dot, inbound_layer_id, layer_id)
                    elif not isinstance(inbound_layer, functional.Functional):
                        if not isinstance(layer, functional.Functional):
                            assert dot.get_node(inbound_layer_id)
                            assert dot.get_node(layer_id)
                            add_edge(dot, inbound_layer_id, layer_id)
                        elif isinstance(layer, functional.Functional):
                            add_edge(dot, inbound_layer_id, sub_n_first_node[layer.name].get_name())
                    elif isinstance(inbound_layer, functional.Functional):
                        name = sub_n_last_node[inbound_layer.name].get_name()
                        if isinstance(layer, functional.Functional):
                            output_name = sub_n_first_node[layer.name].get_name()
                            add_edge(dot, name, output_name)
                        else:
                            add_edge(dot, name, layer_id)
    return dot

@keras_export('keras.utils.plot_model')
def plot_model(model, to_file='model.png', show_shapes=False, show_dtype=False, show_layer_names=False, rankdir='TB', expand_nested=False, dpi=200, show_layer_activations=False, show_trainable=False, **kwargs):
    if False:
        return 10
    'Converts a Keras model to dot format and save to a file.\n\n    Example:\n\n    ```python\n    inputs = ...\n    outputs = ...\n    model = keras.Model(inputs=inputs, outputs=outputs)\n\n    dot_img_file = \'/tmp/model_1.png\'\n    keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)\n    ```\n\n    Args:\n        model: A Keras model instance\n        to_file: File name of the plot image.\n        show_shapes: whether to display shape information.\n        show_dtype: whether to display layer dtypes.\n        show_layer_names: whether to display layer names.\n        rankdir: `rankdir` argument passed to PyDot,\n            a string specifying the format of the plot: `"TB"`\n            creates a vertical plot; `"LR"` creates a horizontal plot.\n        expand_nested: whether to expand nested Functional models\n            into clusters.\n        dpi: Image resolution in dots per inch.\n        show_layer_activations: Display layer activations (only for layers that\n            have an `activation` property).\n        show_trainable: whether to display if a layer is trainable.\n\n    Returns:\n        A Jupyter notebook Image object if Jupyter is installed.\n        This enables in-line display of the model plots in notebooks.\n    '
    if not model.built:
        raise ValueError('This model has not yet been built. Build the model first by calling `build()` or by calling the model on a batch of data.')
    if not check_pydot():
        message = 'You must install pydot (`pip install pydot`) for `plot_model` to work.'
        if 'IPython.core.magics.namespace' in sys.modules:
            io_utils.print_msg(message)
            return
        else:
            raise ImportError(message)
    if not check_graphviz():
        message = 'You must install graphviz (see instructions at https://graphviz.gitlab.io/download/) for `plot_model` to work.'
        if 'IPython.core.magics.namespace' in sys.modules:
            io_utils.print_msg(message)
            return
        else:
            raise ImportError(message)
    if kwargs.pop('layer_range', None) is not None:
        raise ValueError('Argument `layer_range` is no longer supported.')
    if kwargs:
        raise ValueError(f'Unrecognized keyword arguments: {kwargs}')
    dot = model_to_dot(model, show_shapes=show_shapes, show_dtype=show_dtype, show_layer_names=show_layer_names, rankdir=rankdir, expand_nested=expand_nested, dpi=dpi, show_layer_activations=show_layer_activations, show_trainable=show_trainable)
    to_file = str(to_file)
    if dot is None:
        return
    (_, extension) = os.path.splitext(to_file)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(to_file, format=extension)
    if extension != 'pdf':
        try:
            from IPython import display
            return display.Image(filename=to_file)
        except ImportError:
            pass