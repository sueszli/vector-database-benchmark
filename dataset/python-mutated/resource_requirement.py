from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AbstractSet, Iterator, Mapping, NamedTuple, Optional, Sequence, Type
from dagster._utils.merger import merge_dicts
from ..errors import DagsterInvalidDefinitionError, DagsterInvalidInvocationError
from .utils import DEFAULT_IO_MANAGER_KEY
if TYPE_CHECKING:
    from .resource_definition import ResourceDefinition

class ResourceRequirement(ABC):

    @property
    @abstractmethod
    def key(self) -> str:
        if False:
            return 10
        raise NotImplementedError()

    @abstractmethod
    def describe_requirement(self) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @property
    def expected_type(self) -> Type:
        if False:
            return 10
        from .resource_definition import ResourceDefinition
        return ResourceDefinition

    @property
    def is_io_manager_requirement(self) -> bool:
        if False:
            print('Hello World!')
        from ..storage.io_manager import IInputManagerDefinition, IOManagerDefinition
        return self.expected_type == IOManagerDefinition or self.expected_type == IInputManagerDefinition

    def keys_of_expected_type(self, resource_defs: Mapping[str, 'ResourceDefinition']) -> Sequence[str]:
        if False:
            return 10
        'Get resource keys that correspond to resource definitions of expected type.\n\n        For example, if this particular ResourceRequirement subclass required an ``IOManagerDefinition``, this method would vend all keys that corresponded to ``IOManagerDefinition``s.\n        '
        return [resource_key for (resource_key, resource_def) in resource_defs.items() if isinstance(resource_def, self.expected_type)]

    def resource_is_expected_type(self, resource_defs: Mapping[str, 'ResourceDefinition']) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return isinstance(resource_defs[self.key], self.expected_type)

    def resources_contain_key(self, resource_defs: Mapping[str, 'ResourceDefinition']) -> bool:
        if False:
            i = 10
            return i + 15
        return self.key in resource_defs

class ResourceAddable(ABC):

    @abstractmethod
    def with_resources(self, resource_defs: Mapping[str, 'ResourceDefinition']) -> 'ResourceAddable':
        if False:
            return 10
        raise NotImplementedError()

class OpDefinitionResourceRequirement(NamedTuple('_OpDefinitionResourceRequirement', [('key', str), ('node_description', str)]), ResourceRequirement):

    def describe_requirement(self) -> str:
        if False:
            return 10
        return f"resource with key '{self.key}' required by {self.node_description}"

class InputManagerRequirement(NamedTuple('_InputManagerRequirement', [('key', str), ('node_description', str), ('input_name', str), ('root_input', bool)]), ResourceRequirement):

    @property
    def expected_type(self) -> Type:
        if False:
            return 10
        from ..storage.io_manager import IInputManagerDefinition
        return IInputManagerDefinition

    def describe_requirement(self) -> str:
        if False:
            print('Hello World!')
        return f"input manager with key '{self.key}' required by input '{self.input_name}' of {self.node_description}"

class SourceAssetIOManagerRequirement(NamedTuple('_InputManagerRequirement', [('key', str), ('asset_key', Optional[str])]), ResourceRequirement):

    @property
    def expected_type(self) -> Type:
        if False:
            while True:
                i = 10
        from ..storage.io_manager import IOManagerDefinition
        return IOManagerDefinition

    def describe_requirement(self) -> str:
        if False:
            i = 10
            return i + 15
        source_asset_descriptor = f'SourceAsset with key {self.asset_key}' if self.asset_key else 'SourceAsset'
        return f"io manager with key '{self.key}' required by {source_asset_descriptor}"

class OutputManagerRequirement(NamedTuple('_OutputManagerRequirement', [('key', str), ('node_description', str), ('output_name', str)]), ResourceRequirement):

    @property
    def expected_type(self) -> Type:
        if False:
            for i in range(10):
                print('nop')
        from ..storage.io_manager import IOManagerDefinition
        return IOManagerDefinition

    def describe_requirement(self) -> str:
        if False:
            return 10
        return f"io manager with key '{self.key}' required by output '{self.output_name}' of {self.node_description}'"

class HookResourceRequirement(NamedTuple('_HookResourceRequirement', [('key', str), ('attached_to', Optional[str]), ('hook_name', str)]), ResourceRequirement):

    def describe_requirement(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        attached_to_desc = f'attached to {self.attached_to}' if self.attached_to else ''
        return f"resource with key '{self.key}' required by hook '{self.hook_name}' {attached_to_desc}"

class TypeResourceRequirement(NamedTuple('_TypeResourceRequirement', [('key', str), ('type_display_name', str)]), ResourceRequirement):

    def describe_requirement(self) -> str:
        if False:
            print('Hello World!')
        return f"resource with key '{self.key}' required by type '{self.type_display_name}'"

class TypeLoaderResourceRequirement(NamedTuple('_TypeLoaderResourceRequirement', [('key', str), ('type_display_name', str)]), ResourceRequirement):

    def describe_requirement(self) -> str:
        if False:
            return 10
        return f"resource with key '{self.key}' required by the loader on type '{self.type_display_name}'"

class ResourceDependencyRequirement(NamedTuple('_ResourceDependencyRequirement', [('key', str), ('source_key', Optional[str])]), ResourceRequirement):

    def describe_requirement(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        source_descriptor = f" by resource with key '{self.source_key}'" if self.source_key else ''
        return f"resource with key '{self.key}' required{source_descriptor}"

class RequiresResources(ABC):

    @property
    @abstractmethod
    def required_resource_keys(self) -> AbstractSet[str]:
        if False:
            return 10
        raise NotImplementedError()

    @abstractmethod
    def get_resource_requirements(self, outer_context: Optional[object]=None) -> Iterator[ResourceRequirement]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

def ensure_resources_of_expected_type(resource_defs: Mapping[str, 'ResourceDefinition'], requirements: Sequence[ResourceRequirement]) -> None:
    if False:
        return 10
    for requirement in requirements:
        if requirement.resources_contain_key(resource_defs) and (not requirement.resource_is_expected_type(resource_defs)):
            raise DagsterInvalidDefinitionError(f'{requirement.describe_requirement()}, but received {type(resource_defs[requirement.key])}.')

def ensure_requirements_satisfied(resource_defs: Mapping[str, 'ResourceDefinition'], requirements: Sequence[ResourceRequirement]) -> None:
    if False:
        for i in range(10):
            print('nop')
    ensure_resources_of_expected_type(resource_defs, requirements)
    for requirement in requirements:
        if not requirement.resources_contain_key(resource_defs):
            requirement_expected_type_name = requirement.expected_type.__name__
            raise DagsterInvalidDefinitionError(f"{requirement.describe_requirement()} was not provided. Please provide a {requirement_expected_type_name} to key '{requirement.key}', or change the required key to one of the following keys which points to an {requirement_expected_type_name}: {requirement.keys_of_expected_type(resource_defs)}")

def get_resource_key_conflicts(resource_defs: Mapping[str, 'ResourceDefinition'], other_resource_defs: Mapping[str, 'ResourceDefinition']) -> AbstractSet[str]:
    if False:
        return 10
    overlapping_keys = set(resource_defs.keys()).intersection(set(other_resource_defs.keys()))
    overlapping_keys = {key for key in overlapping_keys if key != DEFAULT_IO_MANAGER_KEY}
    return overlapping_keys

def merge_resource_defs(old_resource_defs: Mapping[str, 'ResourceDefinition'], resource_defs_to_merge_in: Mapping[str, 'ResourceDefinition'], requires_resources: RequiresResources) -> Mapping[str, 'ResourceDefinition']:
    if False:
        return 10
    from dagster._core.execution.resources_init import get_transitive_required_resource_keys
    overlapping_keys = get_resource_key_conflicts(old_resource_defs, resource_defs_to_merge_in)
    if overlapping_keys:
        overlapping_keys_str = ', '.join(sorted(list(overlapping_keys)))
        raise DagsterInvalidInvocationError(f"{requires_resources} has conflicting resource definitions with provided resources for the following keys: {overlapping_keys_str}. Either remove the existing resources from the asset or change the resource keys so that they don't overlap.")
    merged_resource_defs = merge_dicts(resource_defs_to_merge_in, old_resource_defs)
    ensure_requirements_satisfied(merged_resource_defs, list(requires_resources.get_resource_requirements()))
    relevant_keys = get_transitive_required_resource_keys(requires_resources.required_resource_keys, merged_resource_defs)
    return {key: resource_def for (key, resource_def) in merged_resource_defs.items() if key in relevant_keys}