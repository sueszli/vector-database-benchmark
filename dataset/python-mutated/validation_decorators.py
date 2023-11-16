"""Decorators for assigning DoFn types to specific storage models."""
from __future__ import annotations
import collections
import inspect
import itertools
import re
from core.jobs import job_utils
from core.jobs.types import base_validation_errors
from core.jobs.types import model_property
from core.platform import models
import apache_beam as beam
from apache_beam import typehints
from typing import Callable, Dict, FrozenSet, Iterator, Sequence, Set, Tuple, Type, cast
MYPY = False
if MYPY:
    from mypy_imports import base_models
(base_models,) = models.Registry.import_models([models.Names.BASE_MODEL])
_ALL_MODEL_TYPES: FrozenSet[Type[base_models.BaseModel]] = frozenset(models.Registry.get_all_storage_model_classes())
_ALL_BASE_MODEL_TYPES: FrozenSet[Type[base_models.BaseModel]] = frozenset(models.Registry.get_storage_model_classes([models.Names.BASE_MODEL]))
_MODEL_TYPES_BY_BASE_CLASS: Dict[Type[base_models.BaseModel], FrozenSet[Type[base_models.BaseModel]]] = {base_class: frozenset({base_class}).union((t for t in _ALL_MODEL_TYPES if issubclass(t, base_class))) for base_class in _ALL_BASE_MODEL_TYPES}
ModelRelationshipsType = Callable[..., Iterator[Tuple[model_property.PropertyType, Sequence[Type[base_models.BaseModel]]]]]

class AuditsExisting:
    """Decorator for registering DoFns that audit storage models.

    DoFns registered by this decorator should assume that the models they
    receive as input do not have `deleted=True`.

    When decorating a DoFn that inherits from another, it overwrites the base
    class. For example, ValidateExplorationModelId overwrites ValidateModelId if
    and only if ValidateExplorationModelId inherits from ValidateModelId.
    """
    _DO_FN_TYPES_BY_KIND: Dict[str, Set[Type[beam.DoFn]]] = collections.defaultdict(set)

    def __init__(self, *model_types: Type[base_models.BaseModel]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initializes the decorator to target the given types of models.\n\n        Args:\n            *model_types: tuple(class). The models the decorator will target. If\n                an argument is a base class, all of its subclasses will be\n                targeted as well.\n\n        Raises:\n            ValueError. No model given.\n            TypeError. When a non-model type is provided.\n        '
        if not model_types:
            raise ValueError('Must target at least one model')
        self._targeted_model_types: Set[Type[base_models.BaseModel]] = set()
        for t in model_types:
            if t in _MODEL_TYPES_BY_BASE_CLASS:
                self._targeted_model_types.update(_MODEL_TYPES_BY_BASE_CLASS[t])
            elif t in _ALL_MODEL_TYPES:
                self._targeted_model_types.add(t)
            else:
                raise TypeError('%r is not a model registered in core.platform' % t)
        self._targeted_kinds = {job_utils.get_model_kind(t) for t in self._targeted_model_types}

    def __call__(self, do_fn_type: Type[beam.DoFn]) -> Type[beam.DoFn]:
        if False:
            return 10
        'Decorator which registers the given DoFn to the targeted models.\n\n        This decorator also installs type constraints on the DoFn to guard it\n        from invalid argument types.\n\n        Args:\n            do_fn_type: type(DoFn). The new audting DoFn class to decorate.\n\n        Returns:\n            type(DoFn). The decorated DoFn.\n\n        Raises:\n            TypeError. When the new type is not a DoFn.\n        '
        if not issubclass(do_fn_type, beam.DoFn):
            raise TypeError('%r is not a subclass of DoFn' % do_fn_type)
        base_types_of_do_fn_type = set(inspect.getmro(do_fn_type))
        for kind in self._targeted_kinds:
            registered_do_fn_types = self._DO_FN_TYPES_BY_KIND[kind]
            if any((issubclass(r, do_fn_type) for r in registered_do_fn_types)):
                continue
            registered_do_fn_types -= base_types_of_do_fn_type
            registered_do_fn_types.add(do_fn_type)
        (with_input_types, with_output_types) = (typehints.with_input_types(typehints.Union[self._targeted_model_types]), typehints.with_output_types(base_validation_errors.BaseAuditError))
        return cast(Type[beam.DoFn], with_input_types(with_output_types(do_fn_type)))

    @classmethod
    def get_audit_do_fn_types_by_kind(cls) -> Dict[str, FrozenSet[Type[beam.DoFn]]]:
        if False:
            print('Hello World!')
        'Returns the sets of audit DoFns targeting a kind of model.\n\n        Returns:\n            dict(str: frozenset(type(DoFn))). DoFn types, keyed by the kind of\n            model they have targeted.\n        '
        return {kind: frozenset(do_fn_types) for (kind, do_fn_types) in cls._DO_FN_TYPES_BY_KIND.items()}

class RelationshipsOf:
    """Decorator for describing {Model.property: Model.ID} relationships.

    This decorator adds a domain-specific language (DSL) for defining the
    relationship between model properties and the IDs of related models.

    The name of the function is enforced by the decorator so that code reads
    intuitively:
        "Relationships Of [MODEL_CLASS]
        "define model_relationships(model):
            yield model.property, [RELATED_MODELS...]

    Example:
        @RelationshipsOf(UserAuthDetailsModel)
        def user_auth_details_model_relationships(model):
            yield (model.id, [UserSettingsModel])
            yield (model.firebase_auth_id, [UserIdByFirebaseAuthId])
            yield (model.gae_id, [UserIdentifiersModel])
    """
    _ID_REFERENCING_PROPERTIES: Dict[model_property.ModelProperty, Set[str]] = collections.defaultdict(set)

    def __init__(self, model_class: Type[base_models.BaseModel]) -> None:
        if False:
            i = 10
            return i + 15
        'Initializes a new RelationshipsOf decorator.\n\n        Args:\n            model_class: class. A subclass of BaseModel.\n        '
        self._model_kind = self._get_model_kind(model_class)
        self._model_class = model_class

    def __call__(self, model_relationships: ModelRelationshipsType) -> ModelRelationshipsType:
        if False:
            for i in range(10):
                print('nop')
        "Registers the property relationships of self._model_kind yielded by\n        the generator.\n\n        See RelationshipsOf's docstring for a usage example.\n\n        Args:\n            model_relationships: callable. Expected to yield tuples of type\n                (Property, list(class)), where the properties are from the\n                argument provided to the function.\n\n        Returns:\n            generator. The same object.\n        "
        self._validate_name_of_model_relationships(model_relationships)
        yielded_items = model_relationships(self._model_class)
        for (property_instance, referenced_models) in yielded_items:
            property_of_model = model_property.ModelProperty(self._model_class, property_instance)
            self._ID_REFERENCING_PROPERTIES[property_of_model].update((self._get_model_kind(m) for m in referenced_models if m is not self._model_class))
        return model_relationships

    @classmethod
    def get_id_referencing_properties_by_kind_of_possessor(cls) -> Dict[str, Tuple[Tuple[model_property.ModelProperty, Tuple[str, ...]], ...]]:
        if False:
            while True:
                i = 10
        'Returns properties whose values refer to the IDs of the corresponding\n        set of model kinds, grouped by the kind of model the properties belong\n        to.\n\n        Returns:\n            dict(str, tuple(tuple(ModelProperty, tuple(str)))). Tuples of\n            (ModelProperty, set(kind of models)), grouped by the kind of model\n            the properties belong to.\n        '
        by_kind: Callable[[model_property.ModelProperty], str] = lambda model_property: model_property.model_kind
        id_referencing_properties_by_kind_of_possessor = itertools.groupby(sorted(cls._ID_REFERENCING_PROPERTIES.keys(), key=by_kind), key=by_kind)
        references_of: Callable[[model_property.ModelProperty], Set[str]] = lambda p: cls._ID_REFERENCING_PROPERTIES[p]
        return {kind: tuple(((p, tuple(references_of(p))) for p in properties)) for (kind, properties) in id_referencing_properties_by_kind_of_possessor}

    @classmethod
    def get_all_model_kinds_referenced_by_properties(cls) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        "Returns all model kinds that are referenced by another's property.\n\n        Returns:\n            set(str). All model kinds referenced by one or more properties,\n            excluding the models' own ID.\n        "
        return set(itertools.chain.from_iterable(cls._ID_REFERENCING_PROPERTIES.values()))

    @classmethod
    def get_model_kind_references(cls, model_kind: str, property_name: str) -> Set[str]:
        if False:
            print('Hello World!')
        "Returns the kinds of models referenced by the given property.\n\n        Args:\n            model_kind: str. The kind of model the property belongs to.\n            property_name: str. The property's name.\n\n        Returns:\n            set(str). The kinds of models referenced by the given property.\n        "
        model_cls = job_utils.get_model_class(model_kind)
        assert issubclass(model_cls, base_models.BaseModel)
        prop = model_property.ModelProperty(model_cls, getattr(model_cls, property_name))
        return cls._ID_REFERENCING_PROPERTIES.get(prop, set())

    def _get_model_kind(self, model_class: Type[base_models.BaseModel]) -> str:
        if False:
            for i in range(10):
                print('nop')
        "Returns the kind of the model class.\n\n        Args:\n            model_class: BaseModel. A subclass of BaseModel.\n\n        Returns:\n            str. The model's kind.\n\n        Raises:\n            TypeError. The model class is not a subclass of BaseModel.\n        "
        if not isinstance(model_class, type):
            raise TypeError('%r is an instance, not a type' % model_class)
        if not issubclass(model_class, base_models.BaseModel):
            raise TypeError('%s is not a subclass of BaseModel' % model_class.__name__)
        return job_utils.get_model_kind(model_class)

    def _validate_name_of_model_relationships(self, model_relationships: ModelRelationshipsType) -> None:
        if False:
            i = 10
            return i + 15
        'Checks that the model_relationships function has the expected name.\n\n        Args:\n            model_relationships: callable. The function to validate.\n\n        Raises:\n            ValueError. The function is named incorrectly.\n        '
        lower_snake_case_model_kind = re.sub('(?<!^)(?=[A-Z])', '_', self._model_kind).lower()
        expected_name = '%s_relationships' % lower_snake_case_model_kind
        actual_name = model_relationships.__name__
        if actual_name != expected_name:
            raise ValueError('Please rename the function from "%s" to "%s"' % (actual_name, expected_name))