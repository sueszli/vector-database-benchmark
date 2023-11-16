from __future__ import annotations
from django.db.models import Model
from sentry.api.serializers.base import registry
from sentry.testutils.silo import validate_models_have_silos, validate_no_cross_silo_deletions, validate_no_cross_silo_foreign_keys
decorator_exemptions: set[type[Model]] = set()
fk_exemptions: set[tuple[type[Model], type[Model]]] = set()

def test_models_have_silos():
    if False:
        print('Hello World!')
    validate_models_have_silos(decorator_exemptions)

def test_silo_foreign_keys():
    if False:
        while True:
            i = 10
    for unused in fk_exemptions - validate_no_cross_silo_foreign_keys(fk_exemptions):
        raise ValueError(f'fk_exemptions includes non conflicting relation {unused!r}')

def test_cross_silo_deletions():
    if False:
        print('Hello World!')
    validate_no_cross_silo_deletions(fk_exemptions)

def test_no_serializers_for_hybrid_cloud_dataclasses():
    if False:
        while True:
            i = 10
    for type in registry.keys():
        if 'hybrid_cloud' in type.__module__:
            raise ValueError(f'{type!r} has a registered serializer, but we should not create serializers for hybrid cloud dataclasses.')