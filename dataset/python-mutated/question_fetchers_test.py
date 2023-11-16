"""Tests for methods defined in question fetchers.."""
from __future__ import annotations
from core import feconf
from core.domain import question_domain
from core.domain import question_fetchers
from core.domain import question_services
from core.domain import translation_domain
from core.domain import user_services
from core.platform import models
from core.tests import test_utils
MYPY = False
if MYPY:
    from mypy_imports import question_models
(question_models,) = models.Registry.import_models([models.Names.QUESTION])

class QuestionFetchersUnitTests(test_utils.GenericTestBase):
    """Tests for question fetchers."""

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.signup(self.EDITOR_EMAIL, self.EDITOR_USERNAME)
        self.signup(self.CURRICULUM_ADMIN_EMAIL, self.CURRICULUM_ADMIN_USERNAME)
        self.admin_id = self.get_user_id_from_email(self.CURRICULUM_ADMIN_EMAIL)
        self.editor_id = self.get_user_id_from_email(self.EDITOR_EMAIL)
        self.set_curriculum_admins([self.CURRICULUM_ADMIN_USERNAME])
        self.admin = user_services.get_user_actions_info(self.admin_id)
        self.editor = user_services.get_user_actions_info(self.editor_id)
        self.save_new_skill('skill_1', self.admin_id, description='Skill Description 1')
        self.save_new_skill('skill_2', self.admin_id, description='Skill Description 2')
        self.question_id = question_services.get_new_question_id()
        self.content_id_generator = translation_domain.ContentIdGenerator()
        self.question = self.save_new_question(self.question_id, self.editor_id, self._create_valid_question_data('ABC', self.content_id_generator), ['skill_1'], self.content_id_generator.next_content_id_index)

    def test_get_questions_and_skill_descriptions_by_skill_ids(self) -> None:
        if False:
            return 10
        question_services.create_new_question_skill_link(self.editor_id, self.question_id, 'skill_1', 0.3)
        (questions, _) = question_fetchers.get_questions_and_skill_descriptions_by_skill_ids(2, ['skill_1'], 0)
        assert questions[0] is not None
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].to_dict(), self.question.to_dict())

    def test_get_no_questions_with_no_skill_ids(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (questions, _) = question_fetchers.get_questions_and_skill_descriptions_by_skill_ids(1, [], 0)
        self.assertEqual(len(questions), 0)

    def test_get_questions_with_multi_skill_ids(self) -> None:
        if False:
            return 10
        question_id_1 = question_services.get_new_question_id()
        content_id_generator = translation_domain.ContentIdGenerator()
        question_1 = self.save_new_question(question_id_1, self.editor_id, self._create_valid_question_data('ABC', content_id_generator), ['skill_1', 'skill_2'], content_id_generator.next_content_id_index)
        question_services.create_new_question_skill_link(self.editor_id, question_id_1, 'skill_1', 0.3)
        question_services.create_new_question_skill_link(self.editor_id, question_id_1, 'skill_2', 0.5)
        (questions, _) = question_fetchers.get_questions_and_skill_descriptions_by_skill_ids(2, ['skill_1', 'skill_2'], 0)
        assert questions[0] is not None
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].to_dict(), question_1.to_dict())

    def test_get_questions_by_ids(self) -> None:
        if False:
            i = 10
            return i + 15
        question_id_2 = question_services.get_new_question_id()
        content_id_generator = translation_domain.ContentIdGenerator()
        self.save_new_question(question_id_2, self.editor_id, self._create_valid_question_data('DEF', content_id_generator), ['skill_1'], content_id_generator.next_content_id_index)
        questions = question_fetchers.get_questions_by_ids([self.question_id, 'invalid_question_id', question_id_2])
        self.assertEqual(len(questions), 3)
        assert questions[0] is not None
        self.assertEqual(questions[0].id, self.question_id)
        self.assertIsNone(questions[1])
        assert questions[2] is not None
        self.assertEqual(questions[2].id, question_id_2)

    def test_cannot_get_question_from_model_with_invalid_schema_version(self) -> None:
        if False:
            i = 10
            return i + 15
        all_question_models = question_models.QuestionModel.get_all()
        question_models.QuestionModel.delete_multi([question_model.id for question_model in all_question_models], feconf.SYSTEM_COMMITTER_ID, '', force_deletion=True)
        all_question_models = question_models.QuestionModel.get_all()
        self.assertEqual(all_question_models.count(), 0)
        question_id = question_services.get_new_question_id()
        content_id_generator = translation_domain.ContentIdGenerator()
        question_model = question_models.QuestionModel(id=question_id, question_state_data=self._create_valid_question_data('ABC', content_id_generator).to_dict(), language_code='en', version=0, next_content_id_index=content_id_generator.next_content_id_index, question_state_data_schema_version=0)
        question_model.commit(self.editor_id, 'question model created', [{'cmd': question_domain.CMD_CREATE_NEW}])
        all_question_models = question_models.QuestionModel.get_all()
        self.assertEqual(all_question_models.count(), 1)
        fetched_question_models = all_question_models.get()
        assert fetched_question_models is not None
        with self.assertRaisesRegex(Exception, 'Sorry, we can only process v25-v%d state schemas at present.' % feconf.CURRENT_STATE_SCHEMA_VERSION):
            question_fetchers.get_question_from_model(fetched_question_models)

    def test_get_question_from_model_with_current_valid_schema_version(self) -> None:
        if False:
            print('Hello World!')
        all_question_models = question_models.QuestionModel.get_all()
        question_models.QuestionModel.delete_multi([question_model.id for question_model in all_question_models], feconf.SYSTEM_COMMITTER_ID, '', force_deletion=True)
        all_question_models = question_models.QuestionModel.get_all()
        self.assertEqual(all_question_models.count(), 0)
        question_id = question_services.get_new_question_id()
        content_id_generator = translation_domain.ContentIdGenerator()
        question_model = question_models.QuestionModel(id=question_id, question_state_data=self._create_valid_question_data('ABC', content_id_generator).to_dict(), language_code='en', version=0, next_content_id_index=content_id_generator.next_content_id_index, question_state_data_schema_version=feconf.CURRENT_STATE_SCHEMA_VERSION)
        question_model.commit(self.editor_id, 'question model created', [{'cmd': question_domain.CMD_CREATE_NEW}])
        all_question_models = question_models.QuestionModel.get_all()
        self.assertEqual(all_question_models.count(), 1)
        fetched_question_models = all_question_models.get()
        assert fetched_question_models is not None
        updated_question_model = question_fetchers.get_question_from_model(fetched_question_models)
        self.assertEqual(updated_question_model.question_state_data_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)

    def test_get_questions_by_ids_with_latest_schema_version(self) -> None:
        if False:
            i = 10
            return i + 15
        question_id = question_services.get_new_question_id()
        self.save_new_question_with_state_data_schema_v27(question_id, self.editor_id, [])
        question = question_fetchers.get_questions_by_ids([question_id])[0]
        assert question is not None
        self.assertEqual(question.question_state_data_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)