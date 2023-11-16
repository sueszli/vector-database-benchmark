try:
    from re import Pattern
except ImportError:
    from typing import Pattern
from typing import Callable, Dict, List, Optional, Union, Tuple
from graphql import GraphQLError
from graphql.validation import ValidationContext, ValidationRule
from graphql.language import DefinitionNode, FieldNode, FragmentDefinitionNode, FragmentSpreadNode, InlineFragmentNode, Node, OperationDefinitionNode
from ..utils.is_introspection_key import is_introspection_key
IgnoreType = Union[Callable[[str], bool], Pattern, str]

def depth_limit_validator(max_depth: int, ignore: Optional[List[IgnoreType]]=None, callback: Optional[Callable[[Dict[str, int]], None]]=None):
    if False:
        return 10

    class DepthLimitValidator(ValidationRule):

        def __init__(self, validation_context: ValidationContext):
            if False:
                i = 10
                return i + 15
            document = validation_context.document
            definitions = document.definitions
            fragments = get_fragments(definitions)
            queries = get_queries_and_mutations(definitions)
            query_depths = {}
            for name in queries:
                query_depths[name] = determine_depth(node=queries[name], fragments=fragments, depth_so_far=0, max_depth=max_depth, context=validation_context, operation_name=name, ignore=ignore)
            if callable(callback):
                callback(query_depths)
            super().__init__(validation_context)
    return DepthLimitValidator

def get_fragments(definitions: Tuple[DefinitionNode, ...]) -> Dict[str, FragmentDefinitionNode]:
    if False:
        i = 10
        return i + 15
    fragments = {}
    for definition in definitions:
        if isinstance(definition, FragmentDefinitionNode):
            fragments[definition.name.value] = definition
    return fragments

def get_queries_and_mutations(definitions: Tuple[DefinitionNode, ...]) -> Dict[str, OperationDefinitionNode]:
    if False:
        i = 10
        return i + 15
    operations = {}
    for definition in definitions:
        if isinstance(definition, OperationDefinitionNode):
            operation = definition.name.value if definition.name else 'anonymous'
            operations[operation] = definition
    return operations

def determine_depth(node: Node, fragments: Dict[str, FragmentDefinitionNode], depth_so_far: int, max_depth: int, context: ValidationContext, operation_name: str, ignore: Optional[List[IgnoreType]]=None) -> int:
    if False:
        for i in range(10):
            print('nop')
    if depth_so_far > max_depth:
        context.report_error(GraphQLError(f"'{operation_name}' exceeds maximum operation depth of {max_depth}.", [node]))
        return depth_so_far
    if isinstance(node, FieldNode):
        should_ignore = is_introspection_key(node.name.value) or is_ignored(node, ignore)
        if should_ignore or not node.selection_set:
            return 0
        return 1 + max(map(lambda selection: determine_depth(node=selection, fragments=fragments, depth_so_far=depth_so_far + 1, max_depth=max_depth, context=context, operation_name=operation_name, ignore=ignore), node.selection_set.selections))
    elif isinstance(node, FragmentSpreadNode):
        return determine_depth(node=fragments[node.name.value], fragments=fragments, depth_so_far=depth_so_far, max_depth=max_depth, context=context, operation_name=operation_name, ignore=ignore)
    elif isinstance(node, (InlineFragmentNode, FragmentDefinitionNode, OperationDefinitionNode)):
        return max(map(lambda selection: determine_depth(node=selection, fragments=fragments, depth_so_far=depth_so_far, max_depth=max_depth, context=context, operation_name=operation_name, ignore=ignore), node.selection_set.selections))
    else:
        raise Exception(f'Depth crawler cannot handle: {node.kind}.')

def is_ignored(node: FieldNode, ignore: Optional[List[IgnoreType]]=None) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if ignore is None:
        return False
    for rule in ignore:
        field_name = node.name.value
        if isinstance(rule, str):
            if field_name == rule:
                return True
        elif isinstance(rule, Pattern):
            if rule.match(field_name):
                return True
        elif callable(rule):
            if rule(field_name):
                return True
        else:
            raise ValueError(f'Invalid ignore option: {rule}.')
    return False