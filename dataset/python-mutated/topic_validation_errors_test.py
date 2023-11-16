"""Unit tests for topic model validator errors."""
from __future__ import annotations
from core.jobs.types import base_validation_errors_test
from core.jobs.types import topic_validation_errors
from core.platform import models
MYPY = False
if MYPY:
    from mypy_imports import topic_models
(topic_models,) = models.Registry.import_models([models.Names.TOPIC])
datastore_services = models.Registry.import_datastore_services()

class ModelCanonicalNameMismatchErrorTests(base_validation_errors_test.AuditErrorsTestBase):

    def test_message(self) -> None:
        if False:
            print('Hello World!')
        model = topic_models.TopicModel(id='test', name='name', url_fragment='name-two', canonical_name='canonical_name', next_subtopic_id=1, language_code='en', subtopic_schema_version=0, story_reference_schema_version=0)
        error = topic_validation_errors.ModelCanonicalNameMismatchError(model)
        self.assertEqual(error.stderr, 'ModelCanonicalNameMismatchError in TopicModel(id="test"): Entity name %s in lowercase does not match canonical name %s' % (model.name, model.canonical_name))