import tree
from keras import backend
from keras import utils
from keras.api_export import keras_export
from keras.layers import Input
from keras.layers import InputLayer
from keras.models.functional import Functional
from keras.models.functional import functional_like_constructor
from keras.models.sequential import Sequential
from keras.saving import serialization_lib

@keras_export('keras.models.clone_model')
def clone_model(model, input_tensors=None, clone_function=None):
    if False:
        print('Hello World!')
    'Clone a Functional or Sequential `Model` instance.\n\n    Model cloning is similar to calling a model on new inputs,\n    except that it creates new layers (and thus new weights) instead\n    of sharing the weights of the existing layers.\n\n    Note that\n    `clone_model` will not preserve the uniqueness of shared objects within the\n    model (e.g. a single variable attached to two distinct layers will be\n    restored as two separate variables).\n\n    Args:\n        model: Instance of `Model`\n            (could be a Functional model or a Sequential model).\n        input_tensors: optional list of input tensors or InputLayer objects\n            to build the model upon. If not provided,\n            new `Input` objects will be created.\n        clone_function: Callable to be used to clone each layer in the target\n            model (except `Input` instances). It takes as argument the\n            layer instance to be cloned, and returns the corresponding layer\n            instance to be used in the model copy. If unspecified, this callable\n            becomes the following serialization/deserialization function:\n            `lambda layer: layer.__class__.from_config(layer.get_config())`.\n            By passing a custom callable, you can customize your copy of the\n            model, e.g. by wrapping certain layers of interest (you might want\n            to replace all `LSTM` instances with equivalent\n            `Bidirectional(LSTM(...))` instances, for example).\n            Defaults to `None`.\n\n    Returns:\n        An instance of `Model` reproducing the behavior\n        of the original model, on top of new inputs tensors,\n        using newly instantiated weights. The cloned model may behave\n        differently from the original model if a custom `clone_function`\n        modifies the layer.\n\n    Examples:\n\n    Basic usage:\n\n    ```python\n    # Create a test Sequential model.\n    model = keras.Sequential([\n        keras.layers.Input(shape=(728,)),\n        keras.layers.Dense(32, activation=\'relu\'),\n        keras.layers.Dense(1, activation=\'sigmoid\'),\n    ])\n    # Create a copy of the test model (with freshly initialized weights).\n    new_model = clone_model(model)\n    ```\n\n    Using a `clone_function` to make a model deterministic by setting the\n    random seed everywhere:\n\n    ```python\n    def clone_function(layer):\n        config = layer.get_config()\n        if "seed" in config:\n            config["seed"] = 1337\n        return layer.__class__.from_config(config)\n\n    new_model = clone_model(model)\n    ```\n\n    Note that subclassed models cannot be cloned by default,\n    since their internal layer structure is not known.\n    To achieve equivalent functionality\n    as `clone_model` in the case of a subclassed model, simply make sure\n    that the model class implements `get_config()`\n    (and optionally `from_config()`), and call:\n\n    ```python\n    new_model = model.__class__.from_config(model.get_config())\n    ```\n\n    In the case of a subclassed model, you cannot using a custom\n    `clone_function`.\n    '
    if isinstance(model, Sequential):
        return _clone_sequential_model(model, input_tensors=input_tensors, clone_function=clone_function)
    if isinstance(model, Functional):
        if utils.is_default(model.get_config) or (clone_function or input_tensors):
            return _clone_functional_model(model, input_tensors=input_tensors, clone_function=clone_function)
    if clone_function or input_tensors:
        raise ValueError(f"Arguments clone_function and input_tensors are only supported for Sequential models or Functional models. Received model of type '{model.__class__.__name__}', with clone_function={clone_function} and input_tensors={input_tensors}")
    config = serialization_lib.serialize_keras_object(model)
    return serialization_lib.deserialize_keras_object(config, custom_objects={model.__class__.__name__: model.__class__})

def _clone_sequential_model(model, input_tensors=None, clone_function=None):
    if False:
        print('Hello World!')
    'Clone a `Sequential` model instance.\n\n    Model cloning is similar to calling a model on new inputs,\n    except that it creates new layers (and thus new weights) instead\n    of sharing the weights of the existing layers.\n\n    Args:\n        model: Instance of `Sequential`.\n        input_tensors: optional list of input tensors\n            to build the model upon. If not provided,\n            placeholders will be created.\n        clone_function: callable to be applied on non-input layers in the model.\n            By default, it clones the layer (without copying the weights).\n\n    Returns:\n        An instance of `Sequential` reproducing the behavior\n        of the original model, on top of new inputs tensors,\n        using newly instantiated weights.\n    '
    if clone_function is None:

        def _clone_layer(layer):
            if False:
                print('Hello World!')
            return layer.__class__.from_config(layer.get_config())
        clone_function = _clone_layer
    if not isinstance(model, Sequential):
        raise ValueError(f'Expected `model` argument to be a `Sequential` model instance. Received: model={model}')
    if not callable(clone_function):
        raise ValueError(f'Expected `clone_function` argument to be a callable. Received: clone_function={clone_function}')
    new_layers = [clone_function(layer) for layer in model.layers]
    if isinstance(model._layers[0], InputLayer):
        ref_input_layer = model._layers[0]
        input_name = ref_input_layer.name
        input_batch_shape = ref_input_layer.batch_shape
        input_dtype = ref_input_layer._dtype
    else:
        input_name = None
        input_dtype = None
        input_batch_shape = None
    if input_tensors:
        if isinstance(input_tensors, (list, tuple)):
            if len(input_tensors) != 1:
                raise ValueError('Argument `input_tensors` must contain a single tensor.')
            input_tensors = input_tensors[0]
        if not isinstance(input_tensors, backend.KerasTensor):
            raise ValueError(f'Argument `input_tensors` must be a KerasTensor. Received invalid value: input_tensors={input_tensors}')
        inputs = Input(tensor=input_tensors, name=input_name)
        new_layers = [inputs] + new_layers
    elif input_batch_shape is not None:
        inputs = Input(tensor=input_tensors, batch_shape=input_batch_shape, dtype=input_dtype, name=input_name)
        new_layers = [inputs] + new_layers
    return Sequential(new_layers, name=model.name, trainable=model.trainable)

def _clone_functional_model(model, input_tensors=None, clone_function=None):
    if False:
        i = 10
        return i + 15
    'Clone a `Functional` model instance.\n\n    Model cloning is similar to calling a model on new inputs,\n    except that it creates new layers (and thus new weights) instead\n    of sharing the weights of the existing layers.\n\n    Input layers are always cloned.\n\n    Args:\n        model: Instance of `Functional`.\n        input_tensors: optional list of input tensors\n            to build the model upon. If not provided,\n            placeholders will be created.\n        clone_function: callable to be applied on non-input layers in the model.\n            By default, it clones the layer (without copying the weights).\n\n    Returns:\n        An instance of `Functional` reproducing the behavior\n        of the original model, on top of new inputs tensors,\n        using newly instantiated weights.\n    '
    if clone_function is None:
        seen = {}

        def _clone_layer(layer):
            if False:
                i = 10
                return i + 15
            if layer in seen:
                return seen[layer]
            new_layer = layer.__class__.from_config(layer.get_config())
            seen[layer] = new_layer
            return new_layer
        clone_function = _clone_layer
    if not callable(clone_function):
        raise ValueError(f'Expected `clone_function` argument to be a callable. Received: clone_function={clone_function}')
    if not isinstance(model, Functional):
        raise ValueError(f'Expected `model` argument to be a Functional Model instance. Received: model={model}')
    if input_tensors is not None:
        if not all((isinstance(x, backend.KerasTensor) for x in tree.flatten(input_tensors))):
            raise ValueError(f'All entries in `input_tensors` must be KerasTensors. Received invalid values: inputs_tensors={input_tensors}')
        try:
            tree.assert_same_structure(input_tensors, model.input)
        except TypeError as e:
            raise ValueError(f'`input_tensors` must have the same structure as model.input\nReference structure: {model.input}\nReceived structure: {input_tensors}') from e
    else:
        input_tensors = tree.map_structure(lambda x: Input(batch_shape=x.shape, dtype=x.dtype, name=x.name), model.input)

    def operation_fn(layer):
        if False:
            for i in range(10):
                print('nop')
        new_layer = clone_function(layer)
        return new_layer
    output_tensors = model._run_through_graph(input_tensors, operation_fn=operation_fn)
    if functional_like_constructor(model.__class__):
        new_model = model.__class__(input_tensors, output_tensors, name=model.name)
    else:
        new_model = Functional(input_tensors, output_tensors, name=model.name)
    return new_model