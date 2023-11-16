from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Hashable, Sequence
from pydantic_core import CoreSchema, core_schema
from ..errors import PydanticUserError
from . import _core_utils
from ._core_utils import NEEDS_APPLY_DISCRIMINATED_UNION_METADATA_KEY, CoreSchemaField, collect_definitions, simplify_schema_references
if TYPE_CHECKING:
    from ..types import Discriminator
CORE_SCHEMA_METADATA_DISCRIMINATOR_PLACEHOLDER_KEY = 'pydantic.internal.union_discriminator'

class MissingDefinitionForUnionRef(Exception):
    """Raised when applying a discriminated union discriminator to a schema
    requires a definition that is not yet defined
    """

    def __init__(self, ref: str) -> None:
        if False:
            print('Hello World!')
        self.ref = ref
        super().__init__(f'Missing definition for ref {self.ref!r}')

def set_discriminator(schema: CoreSchema, discriminator: Any) -> None:
    if False:
        i = 10
        return i + 15
    schema.setdefault('metadata', {})
    metadata = schema.get('metadata')
    assert metadata is not None
    metadata[CORE_SCHEMA_METADATA_DISCRIMINATOR_PLACEHOLDER_KEY] = discriminator

def apply_discriminators(schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
    if False:
        for i in range(10):
            print('nop')
    definitions: dict[str, CoreSchema] | None = None

    def inner(s: core_schema.CoreSchema, recurse: _core_utils.Recurse) -> core_schema.CoreSchema:
        if False:
            for i in range(10):
                print('nop')
        nonlocal definitions
        if 'metadata' in s:
            if s['metadata'].get(NEEDS_APPLY_DISCRIMINATED_UNION_METADATA_KEY, True) is False:
                return s
        s = recurse(s, inner)
        if s['type'] == 'tagged-union':
            return s
        metadata = s.get('metadata', {})
        discriminator = metadata.get(CORE_SCHEMA_METADATA_DISCRIMINATOR_PLACEHOLDER_KEY, None)
        if discriminator is not None:
            if definitions is None:
                definitions = collect_definitions(schema)
            s = apply_discriminator(s, discriminator, definitions)
        return s
    return simplify_schema_references(_core_utils.walk_core_schema(schema, inner))

def apply_discriminator(schema: core_schema.CoreSchema, discriminator: str | Discriminator, definitions: dict[str, core_schema.CoreSchema] | None=None) -> core_schema.CoreSchema:
    if False:
        i = 10
        return i + 15
    "Applies the discriminator and returns a new core schema.\n\n    Args:\n        schema: The input schema.\n        discriminator: The name of the field which will serve as the discriminator.\n        definitions: A mapping of schema ref to schema.\n\n    Returns:\n        The new core schema.\n\n    Raises:\n        TypeError:\n            - If `discriminator` is used with invalid union variant.\n            - If `discriminator` is used with `Union` type with one variant.\n            - If `discriminator` value mapped to multiple choices.\n        MissingDefinitionForUnionRef:\n            If the definition for ref is missing.\n        PydanticUserError:\n            - If a model in union doesn't have a discriminator field.\n            - If discriminator field has a non-string alias.\n            - If discriminator fields have different aliases.\n            - If discriminator field not of type `Literal`.\n    "
    from ..types import Discriminator
    if isinstance(discriminator, Discriminator):
        if isinstance(discriminator.discriminator, str):
            discriminator = discriminator.discriminator
        else:
            return discriminator._convert_schema(schema)
    return _ApplyInferredDiscriminator(discriminator, definitions or {}).apply(schema)

class _ApplyInferredDiscriminator:
    """This class is used to convert an input schema containing a union schema into one where that union is
    replaced with a tagged-union, with all the associated debugging and performance benefits.

    This is done by:
    * Validating that the input schema is compatible with the provided discriminator
    * Introspecting the schema to determine which discriminator values should map to which union choices
    * Handling various edge cases such as 'definitions', 'default', 'nullable' schemas, and more

    I have chosen to implement the conversion algorithm in this class, rather than a function,
    to make it easier to maintain state while recursively walking the provided CoreSchema.
    """

    def __init__(self, discriminator: str, definitions: dict[str, core_schema.CoreSchema]):
        if False:
            print('Hello World!')
        self.discriminator = discriminator
        self.definitions = definitions
        self._discriminator_alias: str | None = None
        self._should_be_nullable = False
        self._is_nullable = False
        self._choices_to_handle: list[core_schema.CoreSchema] = []
        self._tagged_union_choices: dict[Hashable, core_schema.CoreSchema] = {}
        self._used = False

    def apply(self, schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
        if False:
            for i in range(10):
                print('nop')
        "Return a new CoreSchema based on `schema` that uses a tagged-union with the discriminator provided\n        to this class.\n\n        Args:\n            schema: The input schema.\n\n        Returns:\n            The new core schema.\n\n        Raises:\n            TypeError:\n                - If `discriminator` is used with invalid union variant.\n                - If `discriminator` is used with `Union` type with one variant.\n                - If `discriminator` value mapped to multiple choices.\n            ValueError:\n                If the definition for ref is missing.\n            PydanticUserError:\n                - If a model in union doesn't have a discriminator field.\n                - If discriminator field has a non-string alias.\n                - If discriminator fields have different aliases.\n                - If discriminator field not of type `Literal`.\n        "
        self.definitions.update(collect_definitions(schema))
        assert not self._used
        schema = self._apply_to_root(schema)
        if self._should_be_nullable and (not self._is_nullable):
            schema = core_schema.nullable_schema(schema)
        self._used = True
        new_defs = collect_definitions(schema)
        missing_defs = self.definitions.keys() - new_defs.keys()
        if missing_defs:
            schema = core_schema.definitions_schema(schema, [self.definitions[ref] for ref in missing_defs])
        return schema

    def _apply_to_root(self, schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
        if False:
            while True:
                i = 10
        'This method handles the outer-most stage of recursion over the input schema:\n        unwrapping nullable or definitions schemas, and calling the `_handle_choice`\n        method iteratively on the choices extracted (recursively) from the possibly-wrapped union.\n        '
        if schema['type'] == 'nullable':
            self._is_nullable = True
            wrapped = self._apply_to_root(schema['schema'])
            nullable_wrapper = schema.copy()
            nullable_wrapper['schema'] = wrapped
            return nullable_wrapper
        if schema['type'] == 'definitions':
            wrapped = self._apply_to_root(schema['schema'])
            definitions_wrapper = schema.copy()
            definitions_wrapper['schema'] = wrapped
            return definitions_wrapper
        if schema['type'] != 'union':
            schema = core_schema.union_schema([schema])
        choices_schemas = [v[0] if isinstance(v, tuple) else v for v in schema['choices'][::-1]]
        self._choices_to_handle.extend(choices_schemas)
        while self._choices_to_handle:
            choice = self._choices_to_handle.pop()
            self._handle_choice(choice)
        if self._discriminator_alias is not None and self._discriminator_alias != self.discriminator:
            discriminator: str | list[list[str | int]] = [[self.discriminator], [self._discriminator_alias]]
        else:
            discriminator = self.discriminator
        return core_schema.tagged_union_schema(choices=self._tagged_union_choices, discriminator=discriminator, custom_error_type=schema.get('custom_error_type'), custom_error_message=schema.get('custom_error_message'), custom_error_context=schema.get('custom_error_context'), strict=False, from_attributes=True, ref=schema.get('ref'), metadata=schema.get('metadata'), serialization=schema.get('serialization'))

    def _handle_choice(self, choice: core_schema.CoreSchema) -> None:
        if False:
            i = 10
            return i + 15
        'This method handles the "middle" stage of recursion over the input schema.\n        Specifically, it is responsible for handling each choice of the outermost union\n        (and any "coalesced" choices obtained from inner unions).\n\n        Here, "handling" entails:\n        * Coalescing nested unions and compatible tagged-unions\n        * Tracking the presence of \'none\' and \'nullable\' schemas occurring as choices\n        * Validating that each allowed discriminator value maps to a unique choice\n        * Updating the _tagged_union_choices mapping that will ultimately be used to build the TaggedUnionSchema.\n        '
        if choice['type'] == 'none':
            self._should_be_nullable = True
        elif choice['type'] == 'definitions':
            self._handle_choice(choice['schema'])
        elif choice['type'] == 'nullable':
            self._should_be_nullable = True
            self._handle_choice(choice['schema'])
        elif choice['type'] == 'union':
            choices_schemas = [v[0] if isinstance(v, tuple) else v for v in choice['choices'][::-1]]
            self._choices_to_handle.extend(choices_schemas)
        elif choice['type'] == 'definition-ref':
            if choice['schema_ref'] not in self.definitions:
                raise MissingDefinitionForUnionRef(choice['schema_ref'])
            self._handle_choice(self.definitions[choice['schema_ref']])
        elif choice['type'] not in {'model', 'typed-dict', 'tagged-union', 'lax-or-strict', 'dataclass', 'dataclass-args'} and (not _core_utils.is_function_with_inner_schema(choice)):
            raise TypeError(f"{choice['type']!r} is not a valid discriminated union variant; should be a `BaseModel` or `dataclass`")
        else:
            if choice['type'] == 'tagged-union' and self._is_discriminator_shared(choice):
                subchoices = [x for x in choice['choices'].values() if not isinstance(x, (str, int))]
                self._choices_to_handle.extend(subchoices[::-1])
                return
            inferred_discriminator_values = self._infer_discriminator_values_for_choice(choice, source_name=None)
            self._set_unique_choice_for_values(choice, inferred_discriminator_values)

    def _is_discriminator_shared(self, choice: core_schema.TaggedUnionSchema) -> bool:
        if False:
            i = 10
            return i + 15
        'This method returns a boolean indicating whether the discriminator for the `choice`\n        is the same as that being used for the outermost tagged union. This is used to\n        determine whether this TaggedUnionSchema choice should be "coalesced" into the top level,\n        or whether it should be treated as a separate (nested) choice.\n        '
        inner_discriminator = choice['discriminator']
        return inner_discriminator == self.discriminator or (isinstance(inner_discriminator, list) and (self.discriminator in inner_discriminator or [self.discriminator] in inner_discriminator))

    def _infer_discriminator_values_for_choice(self, choice: core_schema.CoreSchema, source_name: str | None) -> list[str | int]:
        if False:
            for i in range(10):
                print('nop')
        'This function recurses over `choice`, extracting all discriminator values that should map to this choice.\n\n        `model_name` is accepted for the purpose of producing useful error messages.\n        '
        if choice['type'] == 'definitions':
            return self._infer_discriminator_values_for_choice(choice['schema'], source_name=source_name)
        elif choice['type'] == 'function-plain':
            raise TypeError(f"{choice['type']!r} is not a valid discriminated union variant; should be a `BaseModel` or `dataclass`")
        elif _core_utils.is_function_with_inner_schema(choice):
            return self._infer_discriminator_values_for_choice(choice['schema'], source_name=source_name)
        elif choice['type'] == 'lax-or-strict':
            return sorted(set(self._infer_discriminator_values_for_choice(choice['lax_schema'], source_name=None) + self._infer_discriminator_values_for_choice(choice['strict_schema'], source_name=None)))
        elif choice['type'] == 'tagged-union':
            values: list[str | int] = []
            subchoices = [x for x in choice['choices'].values() if not isinstance(x, (str, int))]
            for subchoice in subchoices:
                subchoice_values = self._infer_discriminator_values_for_choice(subchoice, source_name=None)
                values.extend(subchoice_values)
            return values
        elif choice['type'] == 'union':
            values = []
            for subchoice in choice['choices']:
                subchoice_schema = subchoice[0] if isinstance(subchoice, tuple) else subchoice
                subchoice_values = self._infer_discriminator_values_for_choice(subchoice_schema, source_name=None)
                values.extend(subchoice_values)
            return values
        elif choice['type'] == 'nullable':
            self._should_be_nullable = True
            return self._infer_discriminator_values_for_choice(choice['schema'], source_name=None)
        elif choice['type'] == 'model':
            return self._infer_discriminator_values_for_choice(choice['schema'], source_name=choice['cls'].__name__)
        elif choice['type'] == 'dataclass':
            return self._infer_discriminator_values_for_choice(choice['schema'], source_name=choice['cls'].__name__)
        elif choice['type'] == 'model-fields':
            return self._infer_discriminator_values_for_model_choice(choice, source_name=source_name)
        elif choice['type'] == 'dataclass-args':
            return self._infer_discriminator_values_for_dataclass_choice(choice, source_name=source_name)
        elif choice['type'] == 'typed-dict':
            return self._infer_discriminator_values_for_typed_dict_choice(choice, source_name=source_name)
        elif choice['type'] == 'definition-ref':
            schema_ref = choice['schema_ref']
            if schema_ref not in self.definitions:
                raise MissingDefinitionForUnionRef(schema_ref)
            return self._infer_discriminator_values_for_choice(self.definitions[schema_ref], source_name=source_name)
        else:
            raise TypeError(f"{choice['type']!r} is not a valid discriminated union variant; should be a `BaseModel` or `dataclass`")

    def _infer_discriminator_values_for_typed_dict_choice(self, choice: core_schema.TypedDictSchema, source_name: str | None=None) -> list[str | int]:
        if False:
            return 10
        'This method just extracts the _infer_discriminator_values_for_choice logic specific to TypedDictSchema\n        for the sake of readability.\n        '
        source = 'TypedDict' if source_name is None else f'TypedDict {source_name!r}'
        field = choice['fields'].get(self.discriminator)
        if field is None:
            raise PydanticUserError(f'{source} needs a discriminator field for key {self.discriminator!r}', code='discriminator-no-field')
        return self._infer_discriminator_values_for_field(field, source)

    def _infer_discriminator_values_for_model_choice(self, choice: core_schema.ModelFieldsSchema, source_name: str | None=None) -> list[str | int]:
        if False:
            i = 10
            return i + 15
        source = 'ModelFields' if source_name is None else f'Model {source_name!r}'
        field = choice['fields'].get(self.discriminator)
        if field is None:
            raise PydanticUserError(f'{source} needs a discriminator field for key {self.discriminator!r}', code='discriminator-no-field')
        return self._infer_discriminator_values_for_field(field, source)

    def _infer_discriminator_values_for_dataclass_choice(self, choice: core_schema.DataclassArgsSchema, source_name: str | None=None) -> list[str | int]:
        if False:
            for i in range(10):
                print('nop')
        source = 'DataclassArgs' if source_name is None else f'Dataclass {source_name!r}'
        for field in choice['fields']:
            if field['name'] == self.discriminator:
                break
        else:
            raise PydanticUserError(f'{source} needs a discriminator field for key {self.discriminator!r}', code='discriminator-no-field')
        return self._infer_discriminator_values_for_field(field, source)

    def _infer_discriminator_values_for_field(self, field: CoreSchemaField, source: str) -> list[str | int]:
        if False:
            return 10
        if field['type'] == 'computed-field':
            return []
        alias = field.get('validation_alias', self.discriminator)
        if not isinstance(alias, str):
            raise PydanticUserError(f'Alias {alias!r} is not supported in a discriminated union', code='discriminator-alias-type')
        if self._discriminator_alias is None:
            self._discriminator_alias = alias
        elif self._discriminator_alias != alias:
            raise PydanticUserError(f'Aliases for discriminator {self.discriminator!r} must be the same (got {alias}, {self._discriminator_alias})', code='discriminator-alias')
        return self._infer_discriminator_values_for_inner_schema(field['schema'], source)

    def _infer_discriminator_values_for_inner_schema(self, schema: core_schema.CoreSchema, source: str) -> list[str | int]:
        if False:
            return 10
        'When inferring discriminator values for a field, we typically extract the expected values from a literal\n        schema. This function does that, but also handles nested unions and defaults.\n        '
        if schema['type'] == 'literal':
            return schema['expected']
        elif schema['type'] == 'union':
            values: list[Any] = []
            for choice in schema['choices']:
                choice_schema = choice[0] if isinstance(choice, tuple) else choice
                choice_values = self._infer_discriminator_values_for_inner_schema(choice_schema, source)
                values.extend(choice_values)
            return values
        elif schema['type'] == 'default':
            return self._infer_discriminator_values_for_inner_schema(schema['schema'], source)
        elif schema['type'] == 'function-after':
            return self._infer_discriminator_values_for_inner_schema(schema['schema'], source)
        elif schema['type'] in {'function-before', 'function-wrap', 'function-plain'}:
            validator_type = repr(schema['type'].split('-')[1])
            raise PydanticUserError(f'Cannot use a mode={validator_type} validator in the discriminator field {self.discriminator!r} of {source}', code='discriminator-validator')
        else:
            raise PydanticUserError(f'{source} needs field {self.discriminator!r} to be of type `Literal`', code='discriminator-needs-literal')

    def _set_unique_choice_for_values(self, choice: core_schema.CoreSchema, values: Sequence[str | int]) -> None:
        if False:
            i = 10
            return i + 15
        'This method updates `self.tagged_union_choices` so that all provided (discriminator) `values` map to the\n        provided `choice`, validating that none of these values already map to another (different) choice.\n        '
        for discriminator_value in values:
            if discriminator_value in self._tagged_union_choices:
                existing_choice = self._tagged_union_choices[discriminator_value]
                if existing_choice != choice:
                    raise TypeError(f'Value {discriminator_value!r} for discriminator {self.discriminator!r} mapped to multiple choices')
            else:
                self._tagged_union_choices[discriminator_value] = choice