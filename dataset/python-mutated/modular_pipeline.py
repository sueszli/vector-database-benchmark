"""Helper to integrate modular pipelines into a master pipeline."""
from __future__ import annotations
import copy
from typing import AbstractSet, Iterable
from kedro.pipeline.node import Node
from kedro.pipeline.pipeline import TRANSCODING_SEPARATOR, Pipeline, _strip_transcoding, _transcode_split

class ModularPipelineError(Exception):
    """Raised when a modular pipeline is not adapted and integrated
    appropriately using the helper.
    """
    pass

def _is_all_parameters(name: str) -> bool:
    if False:
        print('Hello World!')
    return name == 'parameters'

def _is_single_parameter(name: str) -> bool:
    if False:
        i = 10
        return i + 15
    return name.startswith('params:')

def _is_parameter(name: str) -> bool:
    if False:
        return 10
    return _is_single_parameter(name) or _is_all_parameters(name)

def _validate_inputs_outputs(inputs: AbstractSet[str], outputs: AbstractSet[str], pipe: Pipeline) -> None:
    if False:
        return 10
    'Safeguards to ensure that:\n    - parameters are not specified under inputs\n    - inputs are only free inputs\n    - outputs do not contain free inputs\n    '
    inputs = {_strip_transcoding(k) for k in inputs}
    outputs = {_strip_transcoding(k) for k in outputs}
    if any((_is_parameter(i) for i in inputs)):
        raise ModularPipelineError("Parameters should be specified in the 'parameters' argument")
    free_inputs = {_strip_transcoding(i) for i in pipe.inputs()}
    if not inputs <= free_inputs:
        raise ModularPipelineError('Inputs should be free inputs to the pipeline')
    if outputs & free_inputs:
        raise ModularPipelineError("Outputs can't contain free inputs to the pipeline")

def _validate_datasets_exist(inputs: AbstractSet[str], outputs: AbstractSet[str], parameters: AbstractSet[str], pipe: Pipeline) -> None:
    if False:
        while True:
            i = 10
    inputs = {_strip_transcoding(k) for k in inputs}
    outputs = {_strip_transcoding(k) for k in outputs}
    existing = {_strip_transcoding(ds) for ds in pipe.data_sets()}
    non_existent = (inputs | outputs | parameters) - existing
    if non_existent:
        raise ModularPipelineError(f"Failed to map datasets and/or parameters: {', '.join(sorted(non_existent))}")

def _get_dataset_names_mapping(names: str | set[str] | dict[str, str] | None=None) -> dict[str, str]:
    if False:
        print('Hello World!')
    'Take a name or a collection of dataset names\n    and turn it into a mapping from the old dataset names to the provided ones if necessary.\n\n    Args:\n        names: A dataset name or collection of dataset names.\n            When str or set[str] is provided, the listed names will stay\n            the same as they are named in the provided pipeline.\n            When dict[str, str] is provided, current names will be\n            mapped to new names in the resultant pipeline.\n    Returns:\n        A dictionary that maps the old dataset names to the provided ones.\n    Examples:\n        >>> _get_dataset_names_mapping("dataset_name")\n        {"dataset_name": "dataset_name"}  # a str name will stay the same\n        >>> _get_dataset_names_mapping(set(["ds_1", "ds_2"]))\n        {"ds_1": "ds_1", "ds_2": "ds_2"}  # a set[str] of names will stay the same\n        >>> _get_dataset_names_mapping({"ds_1": "new_ds_1_name"})\n        {"ds_1": "new_ds_1_name"}  # a dict[str, str] of names will map key to value\n    '
    if names is None:
        return {}
    if isinstance(names, str):
        return {names: names}
    if isinstance(names, dict):
        return copy.deepcopy(names)
    return {item: item for item in names}

def _normalize_param_name(name: str) -> str:
    if False:
        return 10
    'Make sure that a param name has a `params:` prefix before passing to the node'
    return name if name.startswith('params:') else f'params:{name}'

def _get_param_names_mapping(names: str | set[str] | dict[str, str] | None=None) -> dict[str, str]:
    if False:
        i = 10
        return i + 15
    'Take a parameter or a collection of parameter names\n    and turn it into a mapping from existing parameter names to new ones if necessary.\n    It follows the same rule as `_get_dataset_names_mapping` and\n    prefixes the keys on the resultant dictionary with `params:` to comply with node\'s syntax.\n\n    Args:\n        names: A parameter name or collection of parameter names.\n            When str or set[str] is provided, the listed names will stay\n            the same as they are named in the provided pipeline.\n            When dict[str, str] is provided, current names will be\n            mapped to new names in the resultant pipeline.\n    Returns:\n        A dictionary that maps the old parameter names to the provided ones.\n    Examples:\n        >>> _get_param_names_mapping("param_name")\n        {"params:param_name": "params:param_name"}  # a str name will stay the same\n        >>> _get_param_names_mapping(set(["param_1", "param_2"]))\n        # a set[str] of names will stay the same\n        {"params:param_1": "params:param_1", "params:param_2": "params:param_2"}\n        >>> _get_param_names_mapping({"param_1": "new_name_for_param_1"})\n        # a dict[str, str] of names will map key to valu\n        {"params:param_1": "params:new_name_for_param_1"}\n    '
    params = {}
    for (name, new_name) in _get_dataset_names_mapping(names).items():
        if _is_all_parameters(name):
            params[name] = name
        else:
            param_name = _normalize_param_name(name)
            param_new_name = _normalize_param_name(new_name)
            params[param_name] = param_new_name
    return params

def pipeline(pipe: Iterable[Node | Pipeline] | Pipeline, *, inputs: str | set[str] | dict[str, str] | None=None, outputs: str | set[str] | dict[str, str] | None=None, parameters: str | set[str] | dict[str, str] | None=None, tags: str | Iterable[str] | None=None, namespace: str=None) -> Pipeline:
    if False:
        while True:
            i = 10
    "Create a ``Pipeline`` from a collection of nodes and/or ``Pipeline``\\s.\n\n    Args:\n        pipe: The nodes the ``Pipeline`` will be made of. If you\n            provide pipelines among the list of nodes, those pipelines will\n            be expanded and all their nodes will become part of this\n            new pipeline.\n        inputs: A name or collection of input names to be exposed as connection points\n            to other pipelines upstream. This is optional; if not provided, the\n            pipeline inputs are automatically inferred from the pipeline structure.\n            When str or set[str] is provided, the listed input names will stay\n            the same as they are named in the provided pipeline.\n            When dict[str, str] is provided, current input names will be\n            mapped to new names.\n            Must only refer to the pipeline's free inputs.\n        outputs: A name or collection of names to be exposed as connection points\n            to other pipelines downstream. This is optional; if not provided, the\n            pipeline inputs are automatically inferred from the pipeline structure.\n            When str or set[str] is provided, the listed output names will stay\n            the same as they are named in the provided pipeline.\n            When dict[str, str] is provided, current output names will be\n            mapped to new names.\n            Can refer to both the pipeline's free outputs, as well as\n            intermediate results that need to be exposed.\n        parameters: A name or collection of parameters to namespace.\n            When str or set[str] are provided, the listed parameter names will stay\n            the same as they are named in the provided pipeline.\n            When dict[str, str] is provided, current parameter names will be\n            mapped to new names.\n            The parameters can be specified without the `params:` prefix.\n        tags: Optional set of tags to be applied to all the pipeline nodes.\n        namespace: A prefix to give to all dataset names,\n            except those explicitly named with the `inputs`/`outputs`\n            arguments, and parameter references (`params:` and `parameters`).\n\n    Raises:\n        ModularPipelineError: When inputs, outputs or parameters are incorrectly\n            specified, or they do not exist on the original pipeline.\n        ValueError: When underlying pipeline nodes inputs/outputs are not\n            any of the expected types (str, dict, list, or None).\n\n    Returns:\n        A new ``Pipeline`` object.\n    "
    if isinstance(pipe, Pipeline):
        pipe = Pipeline([pipe], tags=tags)
    else:
        pipe = Pipeline(pipe, tags=tags)
    if not any([inputs, outputs, parameters, namespace]):
        return pipe
    inputs = _get_dataset_names_mapping(inputs)
    outputs = _get_dataset_names_mapping(outputs)
    parameters = _get_param_names_mapping(parameters)
    _validate_datasets_exist(inputs.keys(), outputs.keys(), parameters.keys(), pipe)
    _validate_inputs_outputs(inputs.keys(), outputs.keys(), pipe)
    mapping = {**inputs, **outputs, **parameters}

    def _prefix_dataset(name: str) -> str:
        if False:
            i = 10
            return i + 15
        return f'{namespace}.{name}'

    def _prefix_param(name: str) -> str:
        if False:
            return 10
        (_, param_name) = name.split('params:')
        return f'params:{namespace}.{param_name}'

    def _is_transcode_base_in_mapping(name: str) -> bool:
        if False:
            return 10
        (base_name, _) = _transcode_split(name)
        return base_name in mapping

    def _map_transcode_base(name: str):
        if False:
            for i in range(10):
                print('nop')
        (base_name, transcode_suffix) = _transcode_split(name)
        return TRANSCODING_SEPARATOR.join((mapping[base_name], transcode_suffix))

    def _rename(name: str):
        if False:
            for i in range(10):
                print('nop')
        rules = [(lambda n: n in mapping, lambda n: mapping[n]), (_is_all_parameters, lambda n: n), (_is_transcode_base_in_mapping, _map_transcode_base), (lambda n: bool(namespace) and _is_single_parameter(n), _prefix_param), (lambda n: bool(namespace), _prefix_dataset)]
        for (predicate, processor) in rules:
            if predicate(name):
                return processor(name)
        return name

    def _process_dataset_names(datasets: None | str | list[str] | dict[str, str]) -> None | str | list[str] | dict[str, str]:
        if False:
            print('Hello World!')
        if datasets is None:
            return None
        if isinstance(datasets, str):
            return _rename(datasets)
        if isinstance(datasets, list):
            return [_rename(name) for name in datasets]
        if isinstance(datasets, dict):
            return {key: _rename(value) for (key, value) in datasets.items()}
        raise ValueError(f'Unexpected input {datasets} of type {type(datasets)}')

    def _copy_node(node: Node) -> Node:
        if False:
            for i in range(10):
                print('nop')
        new_namespace = node.namespace
        if namespace:
            new_namespace = f'{namespace}.{node.namespace}' if node.namespace else namespace
        return node._copy(inputs=_process_dataset_names(node._inputs), outputs=_process_dataset_names(node._outputs), namespace=new_namespace)
    new_nodes = [_copy_node(n) for n in pipe.nodes]
    return Pipeline(new_nodes, tags=tags)