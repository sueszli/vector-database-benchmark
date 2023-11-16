from django.apps import apps
from django.core import checks
from django.db import models
from django.test import TestCase
from wagtail.models import LockableMixin, RevisionMixin

class TestLockableMixin(TestCase):

    def tearDown(self):
        if False:
            return 10
        for package in ('wagtailcore', 'wagtail.tests'):
            try:
                for model in ('lockablewithoutrevisionmodel', 'lockableincorrectrevisionmodel', 'lockablewithrevisionmodel'):
                    del apps.all_models[package][model]
            except KeyError:
                pass
        apps.clear_cache()

    def test_lockable_mixin_only(self):
        if False:
            for i in range(10):
                print('nop')

        class LockableWithoutRevisionModel(LockableMixin, models.Model):
            pass
        self.assertEqual(LockableWithoutRevisionModel.check(), [])

    def test_incorrect_revision_mixin_order(self):
        if False:
            i = 10
            return i + 15

        class LockableIncorrectRevisionModel(RevisionMixin, LockableMixin, models.Model):
            pass
        self.assertEqual(LockableIncorrectRevisionModel.check(), [checks.Error('LockableMixin must be applied before RevisionMixin.', hint="Move LockableMixin in the model's base classes before RevisionMixin.", obj=LockableIncorrectRevisionModel, id='wagtailcore.E005')])

    def test_correct_revision_mixin_order(self):
        if False:
            i = 10
            return i + 15

        class LockableWithRevisionModel(LockableMixin, RevisionMixin, models.Model):
            pass
        self.assertEqual(LockableWithRevisionModel.check(), [])