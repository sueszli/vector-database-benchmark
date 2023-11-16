"""Unit tests for jobs.transforms.feedback_validation."""
from __future__ import annotations
from core.jobs import job_test_utils
from core.jobs.decorators import validation_decorators
from core.jobs.transforms.validation import feedback_validation
from core.jobs.types import feedback_validation_errors
from core.platform import models
from core.tests import test_utils
import apache_beam as beam
MYPY = False
if MYPY:
    from mypy_imports import feedback_models
(feedback_models,) = models.Registry.import_models([models.Names.FEEDBACK])

class ValidateEntityTypeTests(job_test_utils.PipelinedTestBase):

    def test_model_with_invalid_entity_type_raises_error(self) -> None:
        if False:
            return 10
        model = feedback_models.GeneralFeedbackThreadModel(id='123', entity_id='123', subject='test_subject', entity_type='invalid', created_on=self.NOW, last_updated=self.NOW)
        output = self.pipeline | beam.Create([model]) | beam.ParDo(feedback_validation.ValidateEntityType())
        self.assert_pcoll_equal(output, [feedback_validation_errors.InvalidEntityTypeError(model)])

    def test_model_with_valid_entity_type_raises_no_error(self) -> None:
        if False:
            return 10
        model = feedback_models.GeneralFeedbackThreadModel(id='123', entity_id='123', subject='test_subject', entity_type='exploration', created_on=self.NOW, last_updated=self.NOW)
        output = self.pipeline | beam.Create([model]) | beam.ParDo(feedback_validation.ValidateEntityType())
        self.assert_pcoll_equal(output, [])

class RelationshipsOfTests(test_utils.TestBase):

    def test_feedback_analytics_model_relationships(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertItemsEqual(validation_decorators.RelationshipsOf.get_model_kind_references('FeedbackAnalyticsModel', 'id'), ['ExplorationModel'])