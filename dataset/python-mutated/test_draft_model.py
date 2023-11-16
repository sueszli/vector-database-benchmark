from django.apps import apps
from django.core import checks
from django.db import models
from django.test import TestCase
from wagtail.models import DraftStateMixin, RevisionMixin

class TestDraftStateMixin(TestCase):

    def tearDown(self):
        if False:
            while True:
                i = 10
        for package in ('wagtailcore', 'wagtail.tests'):
            try:
                for model in ('draftstatewithoutrevisionmodel', 'draftstateincorrectrevisionmodel', 'draftstatewithrevisionmodel'):
                    del apps.all_models[package][model]
            except KeyError:
                pass
        apps.clear_cache()

    def test_missing_revision_mixin(self):
        if False:
            for i in range(10):
                print('nop')

        class DraftStateWithoutRevisionModel(DraftStateMixin, models.Model):
            pass
        self.assertEqual(DraftStateWithoutRevisionModel.check(), [checks.Error('DraftStateMixin requires RevisionMixin to be applied after DraftStateMixin.', hint="Add RevisionMixin to the model's base classes after DraftStateMixin.", obj=DraftStateWithoutRevisionModel, id='wagtailcore.E004')])

    def test_incorrect_revision_mixin_order(self):
        if False:
            print('Hello World!')

        class DraftStateIncorrectRevisionModel(RevisionMixin, DraftStateMixin, models.Model):
            pass
        self.assertEqual(DraftStateIncorrectRevisionModel.check(), [checks.Error('DraftStateMixin requires RevisionMixin to be applied after DraftStateMixin.', hint="Add RevisionMixin to the model's base classes after DraftStateMixin.", obj=DraftStateIncorrectRevisionModel, id='wagtailcore.E004')])

    def test_correct_model(self):
        if False:
            while True:
                i = 10

        class DraftStateWithRevisionModel(DraftStateMixin, RevisionMixin, models.Model):
            pass
        self.assertEqual(DraftStateWithRevisionModel.check(), [])