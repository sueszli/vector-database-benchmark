"""Entry point for accessing the full collection of model auditing DoFns.

This module imports all of the "jobs.transforms.*_audits" modules so that their
AuditsExisting decorators are executed. Doing so ensures that the decorated
DoFns are added to AuditsExisting's internal registry, which we delegate to in
the get_audit_do_fn_types_by_kind() function.

TODO(#11475): Add lint checks that ensure all "jobs.transforms.*_audits" modules
are imported into this file.
"""
from __future__ import annotations
from core.jobs.decorators import validation_decorators
from core.jobs.types import model_property
import apache_beam as beam
from typing import Dict, FrozenSet, Set, Tuple, Type
from core.jobs.transforms.validation import auth_validation
from core.jobs.transforms.validation import base_validation
from core.jobs.transforms.validation import blog_validation
from core.jobs.transforms.validation import collection_validation
from core.jobs.transforms.validation import config_validation
from core.jobs.transforms.validation import exp_validation
from core.jobs.transforms.validation import feedback_validation
from core.jobs.transforms.validation import improvements_validation
from core.jobs.transforms.validation import question_validation
from core.jobs.transforms.validation import skill_validation
from core.jobs.transforms.validation import story_validation
from core.jobs.transforms.validation import subtopic_validation
from core.jobs.transforms.validation import topic_validation
from core.jobs.transforms.validation import user_validation

def get_audit_do_fn_types_by_kind() -> Dict[str, FrozenSet[Type[beam.DoFn]]]:
    if False:
        i = 10
        return i + 15
    'Returns the set of DoFns targeting each kind of model.\n\n    Returns:\n        dict(str: set(DoFn)). DoFn classes, keyed by the kind of model they have\n        targeted.\n    '
    return validation_decorators.AuditsExisting.get_audit_do_fn_types_by_kind()

def get_id_referencing_properties_by_kind_of_possessor() -> Dict[str, Tuple[Tuple[model_property.ModelProperty, Tuple[str, ...]], ...]]:
    if False:
        while True:
            i = 10
    'Returns properties whose values refer to the IDs of the corresponding\n    set of model kinds, grouped by the kind of model the properties belong to.\n\n    Returns:\n        dict(str, tuple(tuple(ModelProperty, tuple(str)))). Tuples of type\n        (ModelProperty, tuple(kind of models)), grouped by the kind of model the\n        properties belong to.\n    '
    return validation_decorators.RelationshipsOf.get_id_referencing_properties_by_kind_of_possessor()

def get_all_model_kinds_referenced_by_properties() -> Set[str]:
    if False:
        return 10
    "Returns all model kinds that are referenced by another model's property.\n\n    Returns:\n        set(str). All model kinds referenced by one or more properties,\n        excluding the models' own ID.\n    "
    return validation_decorators.RelationshipsOf.get_all_model_kinds_referenced_by_properties()