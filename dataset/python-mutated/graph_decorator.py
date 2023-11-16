from functools import update_wrapper
from typing import Any, Callable, Mapping, Optional, Sequence, Union, overload
import dagster._check as check
from dagster._core.decorator_utils import format_docstring_for_description
from ..config import ConfigMapping
from ..graph_definition import GraphDefinition
from ..input import GraphIn, InputDefinition
from ..output import GraphOut, OutputDefinition

class _Graph:
    name: Optional[str]
    description: Optional[str]
    input_defs: Sequence[InputDefinition]
    output_defs: Optional[Sequence[OutputDefinition]]
    ins: Optional[Mapping[str, GraphIn]]
    out: Optional[Union[GraphOut, Mapping[str, GraphOut]]]
    tags: Optional[Mapping[str, str]]
    config_mapping: Optional[ConfigMapping]

    def __init__(self, name: Optional[str]=None, description: Optional[str]=None, input_defs: Optional[Sequence[InputDefinition]]=None, output_defs: Optional[Sequence[OutputDefinition]]=None, ins: Optional[Mapping[str, GraphIn]]=None, out: Optional[Union[GraphOut, Mapping[str, GraphOut]]]=None, tags: Optional[Mapping[str, Any]]=None, config_mapping: Optional[ConfigMapping]=None):
        if False:
            return 10
        self.name = check.opt_str_param(name, 'name')
        self.description = check.opt_str_param(description, 'description')
        self.input_defs = check.opt_sequence_param(input_defs, 'input_defs', of_type=InputDefinition)
        self.did_pass_outputs = output_defs is not None or out is not None
        self.output_defs = check.opt_nullable_sequence_param(output_defs, 'output_defs', of_type=OutputDefinition)
        self.ins = ins
        self.out = out
        self.tags = tags
        self.config_mapping = check.opt_inst_param(config_mapping, 'config_mapping', ConfigMapping)

    def __call__(self, fn: Callable[..., Any]) -> GraphDefinition:
        if False:
            while True:
                i = 10
        check.callable_param(fn, 'fn')
        if not self.name:
            self.name = fn.__name__
        if self.ins is not None:
            input_defs = [inp.to_definition(name) for (name, inp) in self.ins.items()]
        else:
            input_defs = check.opt_list_param(self.input_defs, 'input_defs', of_type=InputDefinition)
        if self.out is None:
            output_defs = self.output_defs
        elif isinstance(self.out, GraphOut):
            output_defs = [self.out.to_definition(name=None)]
        else:
            check.dict_param(self.out, 'out', key_type=str, value_type=GraphOut)
            output_defs = [out.to_definition(name=name) for (name, out) in self.out.items()]
        from dagster._core.definitions.composition import do_composition
        (input_mappings, output_mappings, dependencies, node_defs, config_mapping, positional_inputs, node_input_source_assets) = do_composition(decorator_name='@graph', graph_name=self.name, fn=fn, provided_input_defs=input_defs, provided_output_defs=output_defs, ignore_output_from_composition_fn=False, config_mapping=self.config_mapping)
        graph_def = GraphDefinition(name=self.name, dependencies=dependencies, node_defs=node_defs, description=self.description or format_docstring_for_description(fn), input_mappings=input_mappings, output_mappings=output_mappings, config=config_mapping, positional_inputs=positional_inputs, tags=self.tags, node_input_source_assets=node_input_source_assets)
        update_wrapper(graph_def, fn)
        return graph_def

@overload
def graph(compose_fn: Callable) -> GraphDefinition:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def graph(*, name: Optional[str]=..., description: Optional[str]=..., input_defs: Optional[Sequence[InputDefinition]]=..., output_defs: Optional[Sequence[OutputDefinition]]=..., ins: Optional[Mapping[str, GraphIn]]=..., out: Optional[Union[GraphOut, Mapping[str, GraphOut]]]=..., tags: Optional[Mapping[str, Any]]=..., config: Optional[Union[ConfigMapping, Mapping[str, Any]]]=...) -> _Graph:
    if False:
        i = 10
        return i + 15
    ...

def graph(compose_fn: Optional[Callable]=None, *, name: Optional[str]=None, description: Optional[str]=None, input_defs: Optional[Sequence[InputDefinition]]=None, output_defs: Optional[Sequence[OutputDefinition]]=None, ins: Optional[Mapping[str, GraphIn]]=None, out: Optional[Union[GraphOut, Mapping[str, GraphOut]]]=None, tags: Optional[Mapping[str, Any]]=None, config: Optional[Union[ConfigMapping, Mapping[str, Any]]]=None) -> Union[GraphDefinition, _Graph]:
    if False:
        for i in range(10):
            print('nop')
    "Create an op graph with the specified parameters from the decorated composition function.\n\n    Using this decorator allows you to build up a dependency graph by writing a\n    function that invokes ops (or other graphs) and passes the output to subsequent invocations.\n\n    Args:\n        name (Optional[str]):\n            The name of the op graph. Must be unique within any :py:class:`RepositoryDefinition` containing the graph.\n        description (Optional[str]):\n            A human-readable description of the graph.\n        input_defs (Optional[List[InputDefinition]]):\n            Information about the inputs that this graph maps. Information provided here\n            will be combined with what can be inferred from the function signature, with these\n            explicit InputDefinitions taking precedence.\n\n            Uses of inputs in the body of the decorated composition function will determine\n            the :py:class:`InputMappings <InputMapping>` passed to the underlying\n            :py:class:`GraphDefinition`.\n        output_defs (Optional[List[OutputDefinition]]):\n            Output definitions for the graph. If not provided explicitly, these will be inferred from typehints.\n\n            Uses of these outputs in the body of the decorated composition function, as well as the\n            return value of the decorated function, will be used to infer the appropriate set of\n            :py:class:`OutputMappings <OutputMapping>` for the underlying\n            :py:class:`GraphDefinition`.\n\n            To map multiple outputs, return a dictionary from the composition function.\n        ins (Optional[Dict[str, GraphIn]]):\n            Information about the inputs that this graph maps. Information provided here\n            will be combined with what can be inferred from the function signature, with these\n            explicit GraphIn taking precedence.\n        out (Optional[Union[GraphOut, Dict[str, GraphOut]]]):\n            Information about the outputs that this graph maps. Information provided here will be\n            combined with what can be inferred from the return type signature if the function does\n            not use yield.\n\n            To map multiple outputs, return a dictionary from the composition function.\n       tags (Optional[Dict[str, Any]]): Arbitrary metadata for any execution run of the graph.\n            Values that are not strings will be json encoded and must meet the criteria that\n            `json.loads(json.dumps(value)) == value`.  These tag values may be overwritten by tag\n            values provided at invocation time.\n\n       config (Optional[Union[ConfigMapping], Mapping[str, Any]):\n            Describes how the graph is configured at runtime.\n\n            If a :py:class:`ConfigMapping` object is provided, then the graph takes on the config\n            schema of this object. The mapping will be applied at runtime to generate the config for\n            the graph's constituent nodes.\n\n            If a dictionary is provided, then it will be used as the default run config for the\n            graph. This means it must conform to the config schema of the underlying nodes. Note\n            that the values provided will be viewable and editable in the Dagster UI, so be careful\n            with secrets. its constituent nodes.\n\n            If no value is provided, then the config schema for the graph is the default (derived\n            from the underlying nodes).\n    "
    if compose_fn is not None:
        check.invariant(description is None)
        return _Graph()(compose_fn)
    config_mapping = None
    if config is not None and (not isinstance(config, ConfigMapping)):
        config = check.dict_param(config, 'config', key_type=str)
        config_mapping = ConfigMapping(config_fn=lambda _: config, config_schema=None)
    else:
        config_mapping = config
    return _Graph(name=name, description=description, input_defs=input_defs, output_defs=output_defs, ins=ins, out=out, tags=tags, config_mapping=config_mapping)