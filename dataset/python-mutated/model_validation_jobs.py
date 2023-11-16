"""Audit jobs that validate all of the storage models in the datastore."""
from __future__ import annotations
import collections
from core.jobs import base_jobs
from core.jobs import job_utils
from core.jobs.io import ndb_io
from core.jobs.transforms.validation import base_validation
from core.jobs.transforms.validation import base_validation_registry
from core.jobs.types import base_validation_errors
from core.jobs.types import model_property
from core.platform import models
import apache_beam as beam
from typing import Dict, FrozenSet, Iterable, Iterator, List, Set, Tuple, Type
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import datastore_services
(base_models,) = models.Registry.import_models([models.Names.BASE_MODEL])
datastore_services = models.Registry.import_datastore_services()
AUDIT_DO_FN_TYPES_BY_KIND: Dict[str, FrozenSet[Type[beam.DoFn]]] = base_validation_registry.get_audit_do_fn_types_by_kind()
KIND_BY_INDEX: Tuple[str, ...] = tuple(AUDIT_DO_FN_TYPES_BY_KIND.keys())
ID_REFERENCING_PROPERTIES_BY_KIND_OF_POSSESSOR: Dict[str, Tuple[Tuple[model_property.ModelProperty, Tuple[str, ...]], ...]] = base_validation_registry.get_id_referencing_properties_by_kind_of_possessor()
ALL_MODEL_KINDS_REFERENCED_BY_PROPERTIES: Set[str] = base_validation_registry.get_all_model_kinds_referenced_by_properties()

class ModelKey(collections.namedtuple('ModelKey', ['model_kind', 'model_id'])):
    """Helper class for wrapping a (model kind, model ID) pair."""

    @classmethod
    def from_model(cls, model: base_models.BaseModel) -> ModelKey:
        if False:
            return 10
        'Creates a model key from the given model.\n\n        Args:\n            model: Model. The model to create a key for.\n\n        Returns:\n            ModelKey. The corresponding model key.\n        '
        return cls(model_kind=job_utils.get_model_kind(model), model_id=job_utils.get_model_id(model))

class AuditAllStorageModelsJob(base_jobs.JobBase):
    """Runs a comprehensive audit on every model in the datastore."""

    def run(self) -> beam.PCollection[base_validation_errors.BaseAuditError]:
        if False:
            return 10
        'Returns a PCollection of audit errors aggregated from all models.\n\n        Returns:\n            PCollection. A PCollection of audit errors discovered during the\n            audit.\n        '
        (existing_models, deleted_models) = self.pipeline | 'Get all models' >> ndb_io.GetModels(datastore_services.query_everything()) | 'Partition by model.deleted' >> beam.Partition(lambda model, _: int(model.deleted), 2)
        models_of_kind_by_index = existing_models | 'Split models into parallelizable PCollections' >> beam.Partition(lambda m, _, kinds: kinds.index(job_utils.get_model_kind(m)), len(KIND_BY_INDEX), KIND_BY_INDEX)
        existing_key_count_pcolls = []
        missing_key_error_pcolls = []
        audit_error_pcolls = [deleted_models | 'Apply ValidateDeletedModel on deleted models' >> beam.ParDo(base_validation.ValidateDeletedModel())]
        model_groups = zip(KIND_BY_INDEX, models_of_kind_by_index)
        for (kind, models_of_kind) in model_groups:
            audit_error_pcolls.extend(models_of_kind | ApplyAuditDoFns(kind))
            if kind in ALL_MODEL_KINDS_REFERENCED_BY_PROPERTIES:
                existing_key_count_pcolls.append(models_of_kind | GetExistingModelKeyCounts(kind))
            if kind in ID_REFERENCING_PROPERTIES_BY_KIND_OF_POSSESSOR:
                missing_key_error_pcolls.extend(models_of_kind | GetMissingModelKeyErrors(kind))
        existing_key_counts = existing_key_count_pcolls | 'Flatten PCollections of existing key counts' >> beam.Flatten()
        missing_key_errors = missing_key_error_pcolls | 'Flatten PCollections of missing key errors' >> beam.Flatten()
        audit_error_pcolls.append((existing_key_counts, missing_key_errors) | 'Group counts and errors by key' >> beam.CoGroupByKey() | 'Filter keys without any errors' >> beam.FlatMapTuple(self._get_model_relationship_errors))
        return audit_error_pcolls | 'Combine audit results' >> beam.Flatten()

    def _get_model_relationship_errors(self, unused_join_key: ModelKey, counts_and_errors: Tuple[List[int], List[base_validation_errors.ModelRelationshipError]]) -> List[base_validation_errors.ModelRelationshipError]:
        if False:
            print('Hello World!')
        "Returns errors associated with the given model key if it's missing.\n\n        Args:\n            unused_join_key: ModelKey. The key the counts and errors were joined\n                by.\n            counts_and_errors: tuple(list(int), list(ModelRelationshipError)).\n                The join results. The first element is a list of counts\n                corresponding to the number of keys discovered in the datastore.\n                The second element is the list of errors that should be reported\n                when their sum is 0.\n\n        Returns:\n            list(ModelRelationshipError). A list of errors for the given key.\n            Only non-empty when the sum of counts is 0.\n        "
        (counts, errors) = counts_and_errors
        return errors if sum(counts) == 0 else []

class ApplyAuditDoFns(beam.PTransform):
    """Runs every Audit DoFn targeting the models of a specific kind."""

    def __init__(self, kind: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initializes a new ApplyAuditDoFns instance.\n\n        Args:\n            kind: str. The kind of models this PTransform will receive.\n        '
        super().__init__(label='Apply every Audit DoFn targeting %s' % kind)
        self._kind = kind
        self._do_fn_types = tuple(AUDIT_DO_FN_TYPES_BY_KIND[kind])

    def expand(self, inputs: beam.PCollection[base_models.BaseModel]) -> beam.PCollection[base_validation_errors.BaseAuditError]:
        if False:
            i = 10
            return i + 15
        'Returns audit errors from every Audit DoFn targeting the models.\n\n        This is the method that PTransform requires us to override when\n        implementing custom transforms.\n\n        Args:\n            inputs: PCollection. Models of self._kind, can also contain\n                just one model.\n\n        Returns:\n            iterable(PCollection). A chain of PCollections. Each individual one\n            is the result of a specific DoFn, and is labeled as such.\n        '
        return (inputs | 'Apply %s on %s' % (f.__name__, self._kind) >> beam.ParDo(f()) for f in self._do_fn_types)

class GetExistingModelKeyCounts(beam.PTransform):
    """Returns PCollection of (key, count) pairs for each input model."""

    def __init__(self, kind: str) -> None:
        if False:
            print('Hello World!')
        'Initializes the PTransform.\n\n        Args:\n            kind: str. The kind of model this PTransform will receive.\n        '
        super().__init__(label='Generate (key, count)s for all existing %ss' % kind)
        self._kind = kind

    def expand(self, input_or_inputs: beam.PCollection[base_models.BaseModel]) -> beam.PCollection[Tuple[ModelKey, int]]:
        if False:
            i = 10
            return i + 15
        'Returns a PCollection of (key, count) pairs for each input model.\n\n        Args:\n            input_or_inputs: PCollection. The input models.\n\n        Returns:\n            PCollection. The (ModelKey, int) pairs correponding to the input\n            models and their counts (always 1).\n        '
        return input_or_inputs | 'Generate (key, count) for %ss' % self._kind >> beam.Map(lambda model: (ModelKey.from_model(model), 1))

class GetMissingModelKeyErrors(beam.PTransform):
    """Returns PCollection of (key, error) pairs for each referenced model."""

    def __init__(self, kind: str) -> None:
        if False:
            return 10
        'Initializes the PTransform.\n\n        Args:\n            kind: str. The kind of model this PTransform will receive.\n        '
        super().__init__(label='Generate (key, error)s from the ID properties in %s' % kind)
        self._id_referencing_properties = ID_REFERENCING_PROPERTIES_BY_KIND_OF_POSSESSOR[kind]

    def expand(self, input_or_inputs: beam.PCollection[base_models.BaseModel]) -> Iterable[beam.PCollection[Tuple[ModelKey, base_validation_errors.ModelRelationshipError]]]:
        if False:
            return 10
        'Returns PCollections of (key, error) pairs referenced by the models.\n\n        Args:\n            input_or_inputs: PCollection. The input models.\n\n        Returns:\n            iterable(PCollection). The (ModelKey, ModelRelationshipError) pairs\n            corresponding to the models referenced by the ID properties on the\n            input models, and the error that should be reported when they are\n            missing.\n        '
        return (input_or_inputs | 'Generate errors from %s' % property_of_model >> beam.FlatMap(self._generate_missing_key_errors, property_of_model, referenced_kinds) for (property_of_model, referenced_kinds) in self._id_referencing_properties)

    def _generate_missing_key_errors(self, model: base_models.BaseModel, property_of_model: model_property.ModelProperty, referenced_kinds: Tuple[str, ...]) -> Iterator[Tuple[ModelKey, base_validation_errors.ModelRelationshipError]]:
        if False:
            for i in range(10):
                print('nop')
        "Yields all model keys referenced by the given model's properties.\n\n        Args:\n            model: Model. The input model.\n            property_of_model: ModelProperty. The property that holds the ID(s)\n                of referenced model(s).\n            referenced_kinds: tuple(str). The kinds of models that the property\n                refers to.\n\n        Yields:\n            tuple(ModelKey, ModelRelationshipError). The key for a referenced\n            model and the error to report when the key doesn't exist.\n        "
        for property_value in property_of_model.yield_value_from_model(model):
            if property_value is None:
                continue
            model_id = job_utils.get_model_id(model)
            referenced_id = property_value
            for referenced_kind in referenced_kinds:
                error = base_validation_errors.ModelRelationshipError(property_of_model, model_id, referenced_kind, referenced_id)
                yield (ModelKey(referenced_kind, referenced_id), error)