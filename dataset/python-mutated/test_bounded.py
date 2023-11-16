import pytest
from django.db import models
from sentry.backup.scopes import RelocationScope
from sentry.db.models import BoundedBigIntegerField, BoundedIntegerField, BoundedPositiveIntegerField, Model
from sentry.testutils.cases import TestCase

class DummyModel(Model):
    __relocation_scope__ = RelocationScope.Excluded
    foo = models.CharField(max_length=32)
    normint = BoundedIntegerField(null=True)
    bigint = BoundedBigIntegerField(null=True)
    posint = BoundedPositiveIntegerField(null=True)

    class Meta:
        app_label = 'fixtures'

class ModelTest(TestCase):

    def test_large_int(self):
        if False:
            while True:
                i = 10
        with pytest.raises(AssertionError):
            DummyModel.objects.create(normint=int(9223372036854775807), foo='bar')
        with pytest.raises(AssertionError):
            DummyModel.objects.create(bigint=int(9223372036854775808), foo='bar')
        with pytest.raises(AssertionError):
            DummyModel.objects.create(posint=int(9223372036854775808), foo='bar')