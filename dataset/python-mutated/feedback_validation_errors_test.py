"""Unit tests for feedback model validator errors."""
from __future__ import annotations
from core.jobs.types import base_validation_errors_test
from core.jobs.types import feedback_validation_errors
from core.platform import models
MYPY = False
if MYPY:
    from mypy_imports import feedback_models
(feedback_models,) = models.Registry.import_models([models.Names.FEEDBACK])
datastore_services = models.Registry.import_datastore_services()

class InvalidEntityTypeErrorTests(base_validation_errors_test.AuditErrorsTestBase):

    def test_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        model = feedback_models.GeneralFeedbackThreadModel(id='123', entity_id='123', subject='test_subject', entity_type='invalid', created_on=self.NOW, last_updated=self.NOW)
        error = feedback_validation_errors.InvalidEntityTypeError(model)
        self.assertEqual(error.stderr, 'InvalidEntityTypeError in GeneralFeedbackThreadModel(id="123"): entity type %s is invalid.' % model.entity_type)