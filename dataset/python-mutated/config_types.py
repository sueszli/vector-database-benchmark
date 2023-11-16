from typing import List, Optional, Union
import dagster._check as check
import graphene
from dagster._config import ConfigTypeKind, get_recursive_type_keys
from dagster._core.snap import ConfigFieldSnap, ConfigSchemaSnapshot, ConfigTypeSnap
from .util import ResolveInfo, non_null_list
GrapheneConfigTypeUnion = Union['GrapheneEnumConfigType', 'GrapheneCompositeConfigType', 'GrapheneArrayConfigType', 'GrapheneMapConfigType', 'GrapheneNullableConfigType', 'GrapheneRegularConfigType', 'GrapheneScalarUnionConfigType']

def to_config_type(config_schema_snapshot: ConfigSchemaSnapshot, config_type_key: str) -> GrapheneConfigTypeUnion:
    if False:
        return 10
    check.inst_param(config_schema_snapshot, 'config_schema_snapshot', ConfigSchemaSnapshot)
    check.str_param(config_type_key, 'config_type_key')
    config_type_snap = config_schema_snapshot.get_config_snap(config_type_key)
    kind = config_type_snap.kind
    if kind == ConfigTypeKind.ENUM:
        return GrapheneEnumConfigType(config_schema_snapshot, config_type_snap)
    elif ConfigTypeKind.has_fields(kind):
        return GrapheneCompositeConfigType(config_schema_snapshot, config_type_snap)
    elif kind == ConfigTypeKind.ARRAY:
        return GrapheneArrayConfigType(config_schema_snapshot, config_type_snap)
    elif kind == ConfigTypeKind.MAP:
        return GrapheneMapConfigType(config_schema_snapshot, config_type_snap)
    elif kind == ConfigTypeKind.NONEABLE:
        return GrapheneNullableConfigType(config_schema_snapshot, config_type_snap)
    elif kind == ConfigTypeKind.ANY or kind == ConfigTypeKind.SCALAR:
        return GrapheneRegularConfigType(config_schema_snapshot, config_type_snap)
    elif kind == ConfigTypeKind.SCALAR_UNION:
        return GrapheneScalarUnionConfigType(config_schema_snapshot, config_type_snap)
    else:
        check.failed('Should never reach')

def _ctor_kwargs_for_snap(config_type_snap):
    if False:
        i = 10
        return i + 15
    return dict(key=config_type_snap.key, description=config_type_snap.description, is_selector=config_type_snap.kind == ConfigTypeKind.SELECTOR, type_param_keys=config_type_snap.type_param_keys or [])

class GrapheneConfigType(graphene.Interface):
    key = graphene.NonNull(graphene.String)
    description = graphene.String()
    recursive_config_types = graphene.Field(non_null_list(lambda : GrapheneConfigType), description='\nThis is an odd and problematic field. It recursively goes down to\nget all the types contained within a type. The case where it is horrible\nare dictionaries and it recurses all the way down to the leaves. This means\nthat in a case where one is fetching all the types and then all the inner\ntypes keys for those types, we are returning O(N^2) type keys, which\ncan cause awful performance for large schemas. When you have access\nto *all* the types, you should instead only use the type_param_keys\nfield for closed generic types and manually navigate down the to\nfield types client-side.\n\nWhere it is useful is when you are fetching types independently and\nwant to be able to render them, but without fetching the entire schema.\n\nWe use this capability when rendering the sidebar.\n    ')
    type_param_keys = graphene.Field(non_null_list(graphene.String), description='\nThis returns the keys for type parameters of any closed generic type,\n(e.g. List, Optional). This should be used for reconstructing and\nnavigating the full schema client-side and not innerTypes.\n    ')
    is_selector = graphene.NonNull(graphene.Boolean)

    class Meta:
        name = 'ConfigType'

class GrapheneRegularConfigType(graphene.ObjectType):

    class Meta:
        interfaces = (GrapheneConfigType,)
        description = 'Regular is an odd name in this context. It really means Scalar or Any.'
        name = 'RegularConfigType'
    given_name = graphene.NonNull(graphene.String)

    def __init__(self, config_schema_snapshot: ConfigSchemaSnapshot, config_type_snap: ConfigTypeSnap):
        if False:
            i = 10
            return i + 15
        self._config_type_snap = check.inst_param(config_type_snap, 'config_type_snap', ConfigTypeSnap)
        self._config_schema_snapshot = check.inst_param(config_schema_snapshot, 'config_schema_snapshot', ConfigSchemaSnapshot)
        super().__init__(**_ctor_kwargs_for_snap(config_type_snap))

    def resolve_recursive_config_types(self, _graphene_info: ResolveInfo) -> List[GrapheneConfigTypeUnion]:
        if False:
            print('Hello World!')
        return list(map(lambda key: to_config_type(self._config_schema_snapshot, key), get_recursive_type_keys(self._config_type_snap, self._config_schema_snapshot)))

    def resolve_given_name(self, _):
        if False:
            return 10
        return self._config_type_snap.given_name

class GrapheneMapConfigType(graphene.ObjectType):
    key_type = graphene.Field(graphene.NonNull(GrapheneConfigType))
    value_type = graphene.Field(graphene.NonNull(GrapheneConfigType))
    key_label_name = graphene.Field(graphene.String)

    class Meta:
        interfaces = (GrapheneConfigType,)
        name = 'MapConfigType'

    def __init__(self, config_schema_snapshot: ConfigSchemaSnapshot, config_type_snap: ConfigTypeSnap):
        if False:
            while True:
                i = 10
        self._config_type_snap = check.inst_param(config_type_snap, 'config_type_snap', ConfigTypeSnap)
        self._config_schema_snapshot = check.inst_param(config_schema_snapshot, 'config_schema_snapshot', ConfigSchemaSnapshot)
        super().__init__(**_ctor_kwargs_for_snap(config_type_snap))

    def resolve_recursive_config_types(self, _graphene_info: ResolveInfo) -> List[GrapheneConfigTypeUnion]:
        if False:
            i = 10
            return i + 15
        return list(map(lambda key: to_config_type(self._config_schema_snapshot, key), get_recursive_type_keys(self._config_type_snap, self._config_schema_snapshot)))

    def resolve_key_type(self, _graphene_info: ResolveInfo) -> GrapheneConfigTypeUnion:
        if False:
            for i in range(10):
                print('nop')
        return to_config_type(self._config_schema_snapshot, self._config_type_snap.key_type_key)

    def resolve_value_type(self, _graphene_info: ResolveInfo) -> GrapheneConfigTypeUnion:
        if False:
            i = 10
            return i + 15
        return to_config_type(self._config_schema_snapshot, self._config_type_snap.inner_type_key)

    def resolve_key_label_name(self, _graphene_info: ResolveInfo) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._config_type_snap.given_name

class GrapheneWrappingConfigType(graphene.Interface):
    of_type = graphene.Field(graphene.NonNull(GrapheneConfigType))

    class Meta:
        name = 'WrappingConfigType'

class GrapheneArrayConfigType(graphene.ObjectType):

    class Meta:
        interfaces = (GrapheneConfigType, GrapheneWrappingConfigType)
        name = 'ArrayConfigType'

    def __init__(self, config_schema_snapshot: ConfigSchemaSnapshot, config_type_snap: ConfigTypeSnap):
        if False:
            return 10
        self._config_type_snap = check.inst_param(config_type_snap, 'config_type_snap', ConfigTypeSnap)
        self._config_schema_snapshot = check.inst_param(config_schema_snapshot, 'config_schema_snapshot', ConfigSchemaSnapshot)
        super().__init__(**_ctor_kwargs_for_snap(config_type_snap))

    def resolve_recursive_config_types(self, _graphene_info: ResolveInfo) -> List[GrapheneConfigTypeUnion]:
        if False:
            i = 10
            return i + 15
        return list(map(lambda key: to_config_type(self._config_schema_snapshot, key), get_recursive_type_keys(self._config_type_snap, self._config_schema_snapshot)))

    def resolve_of_type(self, _graphene_info: ResolveInfo) -> GrapheneConfigTypeUnion:
        if False:
            while True:
                i = 10
        return to_config_type(self._config_schema_snapshot, self._config_type_snap.inner_type_key)

class GrapheneScalarUnionConfigType(graphene.ObjectType):
    scalar_type = graphene.NonNull(GrapheneConfigType)
    non_scalar_type = graphene.NonNull(GrapheneConfigType)
    scalar_type_key = graphene.NonNull(graphene.String)
    non_scalar_type_key = graphene.NonNull(graphene.String)

    class Meta:
        interfaces = (GrapheneConfigType,)
        name = 'ScalarUnionConfigType'

    def __init__(self, config_schema_snapshot: ConfigSchemaSnapshot, config_type_snap: ConfigTypeSnap):
        if False:
            i = 10
            return i + 15
        self._config_type_snap = check.inst_param(config_type_snap, 'config_type_snap', ConfigTypeSnap)
        self._config_schema_snapshot = check.inst_param(config_schema_snapshot, 'config_schema_snapshot', ConfigSchemaSnapshot)
        super().__init__(**_ctor_kwargs_for_snap(config_type_snap))

    def resolve_recursive_config_types(self, _graphene_info: ResolveInfo) -> List[GrapheneConfigTypeUnion]:
        if False:
            while True:
                i = 10
        return list(map(lambda key: to_config_type(self._config_schema_snapshot, key), get_recursive_type_keys(self._config_type_snap, self._config_schema_snapshot)))

    def get_scalar_type_key(self) -> str:
        if False:
            while True:
                i = 10
        return self._config_type_snap.scalar_type_key

    def get_non_scalar_type_key(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._config_type_snap.non_scalar_type_key

    def resolve_scalar_type_key(self, _) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.get_scalar_type_key()

    def resolve_non_scalar_type_key(self, _) -> str:
        if False:
            print('Hello World!')
        return self.get_non_scalar_type_key()

    def resolve_scalar_type(self, _) -> GrapheneConfigTypeUnion:
        if False:
            return 10
        return to_config_type(self._config_schema_snapshot, self.get_scalar_type_key())

    def resolve_non_scalar_type(self, _) -> GrapheneConfigTypeUnion:
        if False:
            i = 10
            return i + 15
        return to_config_type(self._config_schema_snapshot, self.get_non_scalar_type_key())

class GrapheneNullableConfigType(graphene.ObjectType):

    class Meta:
        interfaces = (GrapheneConfigType, GrapheneWrappingConfigType)
        name = 'NullableConfigType'

    def __init__(self, config_schema_snapshot: ConfigSchemaSnapshot, config_type_snap: ConfigTypeSnap):
        if False:
            print('Hello World!')
        self._config_type_snap = check.inst_param(config_type_snap, 'config_type_snap', ConfigTypeSnap)
        self._config_schema_snapshot = check.inst_param(config_schema_snapshot, 'config_schema_snapshot', ConfigSchemaSnapshot)
        super().__init__(**_ctor_kwargs_for_snap(config_type_snap))

    def resolve_recursive_config_types(self, _graphene_info: ResolveInfo) -> List[GrapheneConfigTypeUnion]:
        if False:
            while True:
                i = 10
        return list(map(lambda key: to_config_type(self._config_schema_snapshot, key), get_recursive_type_keys(self._config_type_snap, self._config_schema_snapshot)))

    def resolve_of_type(self, _graphene_info: ResolveInfo) -> GrapheneConfigTypeUnion:
        if False:
            i = 10
            return i + 15
        return to_config_type(self._config_schema_snapshot, self._config_type_snap.inner_type_key)

class GrapheneEnumConfigValue(graphene.ObjectType):
    value = graphene.NonNull(graphene.String)
    description = graphene.String()

    class Meta:
        name = 'EnumConfigValue'

class GrapheneEnumConfigType(graphene.ObjectType):

    class Meta:
        interfaces = (GrapheneConfigType,)
        name = 'EnumConfigType'
    values = non_null_list(GrapheneEnumConfigValue)
    given_name = graphene.NonNull(graphene.String)

    def __init__(self, config_schema_snapshot: ConfigSchemaSnapshot, config_type_snap: ConfigTypeSnap):
        if False:
            print('Hello World!')
        self._config_type_snap = check.inst_param(config_type_snap, 'config_type_snap', ConfigTypeSnap)
        self._config_schema_snapshot = check.inst_param(config_schema_snapshot, 'config_schema_snapshot', ConfigSchemaSnapshot)
        super().__init__(**_ctor_kwargs_for_snap(config_type_snap))

    def resolve_recursive_config_types(self, _graphene_info: ResolveInfo) -> List[GrapheneConfigTypeUnion]:
        if False:
            for i in range(10):
                print('nop')
        return list(map(lambda key: to_config_type(self._config_schema_snapshot, key), get_recursive_type_keys(self._config_type_snap, self._config_schema_snapshot)))

    def resolve_values(self, _graphene_info: ResolveInfo) -> List[GrapheneEnumConfigValue]:
        if False:
            i = 10
            return i + 15
        return [GrapheneEnumConfigValue(value=ev.value, description=ev.description) for ev in check.not_none(self._config_type_snap.enum_values)]

    def resolve_given_name(self, _):
        if False:
            i = 10
            return i + 15
        return self._config_type_snap.given_name

class GrapheneConfigTypeField(graphene.ObjectType):
    name = graphene.NonNull(graphene.String)
    description = graphene.String()
    config_type = graphene.NonNull(GrapheneConfigType)
    config_type_key = graphene.NonNull(graphene.String)
    is_required = graphene.NonNull(graphene.Boolean)
    default_value_as_json = graphene.String()

    class Meta:
        name = 'ConfigTypeField'

    def resolve_config_type_key(self, _) -> str:
        if False:
            while True:
                i = 10
        return self._field_snap.type_key

    def __init__(self, config_schema_snapshot: ConfigSchemaSnapshot, field_snap: ConfigFieldSnap):
        if False:
            for i in range(10):
                print('nop')
        self._config_schema_snapshot = check.inst_param(config_schema_snapshot, 'config_schema_snapshot', ConfigSchemaSnapshot)
        self._field_snap: ConfigFieldSnap = check.inst_param(field_snap, 'field_snap', ConfigFieldSnap)
        super().__init__(name=field_snap.name, description=field_snap.description, is_required=field_snap.is_required)

    def resolve_config_type(self, _graphene_info: ResolveInfo) -> GrapheneConfigTypeUnion:
        if False:
            return 10
        return to_config_type(self._config_schema_snapshot, self._field_snap.type_key)

    def resolve_default_value_as_json(self, _graphene_info: ResolveInfo) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self._field_snap.default_value_as_json_str

class GrapheneCompositeConfigType(graphene.ObjectType):
    fields = non_null_list(GrapheneConfigTypeField)

    class Meta:
        interfaces = (GrapheneConfigType,)
        name = 'CompositeConfigType'

    def __init__(self, config_schema_snapshot: ConfigSchemaSnapshot, config_type_snap: ConfigTypeSnap):
        if False:
            print('Hello World!')
        self._config_type_snap = check.inst_param(config_type_snap, 'config_type_snap', ConfigTypeSnap)
        self._config_schema_snapshot = check.inst_param(config_schema_snapshot, 'config_schema_snapshot', ConfigSchemaSnapshot)
        super().__init__(**_ctor_kwargs_for_snap(config_type_snap))

    def resolve_recursive_config_types(self, _graphene_info: ResolveInfo) -> List[GrapheneConfigTypeUnion]:
        if False:
            return 10
        return list(map(lambda key: to_config_type(self._config_schema_snapshot, key), get_recursive_type_keys(self._config_type_snap, self._config_schema_snapshot)))

    def resolve_fields(self, _graphene_info: ResolveInfo) -> List[GrapheneConfigTypeField]:
        if False:
            print('Hello World!')
        return sorted([GrapheneConfigTypeField(config_schema_snapshot=self._config_schema_snapshot, field_snap=field_snap) for field_snap in self._config_type_snap.fields or []], key=lambda field: field.name)
types = [GrapheneArrayConfigType, GrapheneCompositeConfigType, GrapheneConfigType, GrapheneConfigTypeField, GrapheneEnumConfigType, GrapheneEnumConfigValue, GrapheneNullableConfigType, GrapheneRegularConfigType, GrapheneScalarUnionConfigType, GrapheneWrappingConfigType, GrapheneMapConfigType]