"""Unit tests for jobs.batch_jobs.exp_recommendation_computation_jobs."""
from __future__ import annotations
import datetime
from core.constants import constants
from core.domain import recommendations_services
from core.jobs import job_test_utils
from core.jobs.batch_jobs import exp_recommendation_computation_jobs
from core.jobs.types import job_run_result
from core.platform import models
from typing import Dict, Final, List, Tuple, Type, Union
MYPY = False
if MYPY:
    from mypy_imports import exp_models
    from mypy_imports import recommendations_models
(exp_models, recommendations_models) = models.Registry.import_models([models.Names.EXPLORATION, models.Names.RECOMMENDATIONS])
StatsType = List[Tuple[str, List[Dict[str, Union[bool, int, str]]]]]

class ComputeExplorationRecommendationsJobTests(job_test_utils.JobTestBase):
    JOB_CLASS: Type[exp_recommendation_computation_jobs.ComputeExplorationRecommendationsJob] = exp_recommendation_computation_jobs.ComputeExplorationRecommendationsJob
    EXP_1_ID: Final = 'exp_1_id'
    EXP_2_ID: Final = 'exp_2_id'
    EXP_3_ID: Final = 'exp_3_id'

    def test_empty_storage(self) -> None:
        if False:
            return 10
        self.assert_job_output_is_empty()

    def test_does_nothing_when_only_one_exploration_exists(self) -> None:
        if False:
            i = 10
            return i + 15
        exp_summary = self.create_model(exp_models.ExpSummaryModel, id=self.EXP_1_ID, deleted=False, title='title', category='category', objective='objective', language_code='lang', community_owned=False, status=constants.ACTIVITY_STATUS_PUBLIC, exploration_model_last_updated=datetime.datetime.utcnow())
        exp_summary.update_timestamps()
        exp_summary.put()
        self.assert_job_output_is_empty()
        exp_recommendations_model = recommendations_models.ExplorationRecommendationsModel.get(self.EXP_1_ID, strict=False)
        self.assertIsNone(exp_recommendations_model)

    def test_creates_recommendations_for_similar_explorations(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        recommendations_services.create_default_topic_similarities()
        exp_summary_1 = self.create_model(exp_models.ExpSummaryModel, id=self.EXP_1_ID, deleted=False, title='title', category='Architecture', objective='objective', language_code='lang', community_owned=False, status=constants.ACTIVITY_STATUS_PUBLIC, exploration_model_last_updated=datetime.datetime.utcnow())
        exp_summary_1.update_timestamps()
        exp_summary_2 = self.create_model(exp_models.ExpSummaryModel, id=self.EXP_2_ID, deleted=False, title='title', category='Architecture', objective='objective', language_code='lang', community_owned=False, status=constants.ACTIVITY_STATUS_PUBLIC, exploration_model_last_updated=datetime.datetime.utcnow())
        exp_summary_2.update_timestamps()
        self.put_multi([exp_summary_1, exp_summary_2])
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='SUCCESS: 2')])
        exp_recommendations_model_1 = recommendations_models.ExplorationRecommendationsModel.get(self.EXP_1_ID)
        assert exp_recommendations_model_1 is not None
        self.assertEqual(exp_recommendations_model_1.recommended_exploration_ids, [self.EXP_2_ID])
        exp_recommendations_model_2 = recommendations_models.ExplorationRecommendationsModel.get(self.EXP_2_ID)
        assert exp_recommendations_model_2 is not None
        self.assertEqual(exp_recommendations_model_2.recommended_exploration_ids, [self.EXP_1_ID])

    def test_skips_private_explorations(self) -> None:
        if False:
            while True:
                i = 10
        recommendations_services.create_default_topic_similarities()
        exp_summary_1 = self.create_model(exp_models.ExpSummaryModel, id=self.EXP_1_ID, deleted=False, title='title', category='Architecture', objective='objective', language_code='lang', community_owned=False, status=constants.ACTIVITY_STATUS_PRIVATE, exploration_model_last_updated=datetime.datetime.utcnow())
        exp_summary_1.update_timestamps()
        exp_summary_2 = self.create_model(exp_models.ExpSummaryModel, id=self.EXP_2_ID, deleted=False, title='title', category='Architecture', objective='objective', language_code='lang', community_owned=False, status=constants.ACTIVITY_STATUS_PRIVATE, exploration_model_last_updated=datetime.datetime.utcnow())
        exp_summary_2.update_timestamps()
        self.put_multi([exp_summary_1, exp_summary_2])
        self.assert_job_output_is_empty()
        exp_recommendations_model_1 = recommendations_models.ExplorationRecommendationsModel.get(self.EXP_1_ID, strict=False)
        self.assertIsNone(exp_recommendations_model_1)
        exp_recommendations_model_2 = recommendations_models.ExplorationRecommendationsModel.get(self.EXP_2_ID, strict=False)
        self.assertIsNone(exp_recommendations_model_2)

    def test_does_not_create_recommendations_for_different_explorations(self) -> None:
        if False:
            while True:
                i = 10
        recommendations_services.create_default_topic_similarities()
        exp_summary_1 = self.create_model(exp_models.ExpSummaryModel, id=self.EXP_1_ID, deleted=False, title='title', category='Architecture', objective='objective', language_code='lang1', community_owned=False, status=constants.ACTIVITY_STATUS_PUBLIC, exploration_model_last_updated=datetime.datetime.utcnow())
        exp_summary_1.update_timestamps()
        exp_summary_2 = self.create_model(exp_models.ExpSummaryModel, id=self.EXP_2_ID, deleted=False, title='title', category='Sport', objective='objective', language_code='lang2', community_owned=False, status=constants.ACTIVITY_STATUS_PUBLIC, exploration_model_last_updated=datetime.datetime.utcnow())
        exp_summary_2.update_timestamps()
        self.put_multi([exp_summary_1, exp_summary_2])
        self.assert_job_output_is_empty()
        exp_recommendations_model_1 = recommendations_models.ExplorationRecommendationsModel.get(self.EXP_1_ID, strict=False)
        self.assertIsNone(exp_recommendations_model_1)
        exp_recommendations_model_2 = recommendations_models.ExplorationRecommendationsModel.get(self.EXP_2_ID, strict=False)
        self.assertIsNone(exp_recommendations_model_2)

    def test_creates_recommendations_for_three_explorations(self) -> None:
        if False:
            return 10
        recommendations_services.create_default_topic_similarities()
        exp_summary_1 = self.create_model(exp_models.ExpSummaryModel, id=self.EXP_1_ID, deleted=False, title='title', category='Architecture', objective='objective', language_code='lang1', community_owned=False, status=constants.ACTIVITY_STATUS_PUBLIC, exploration_model_last_updated=datetime.datetime.utcnow())
        exp_summary_1.update_timestamps()
        exp_summary_2 = self.create_model(exp_models.ExpSummaryModel, id=self.EXP_2_ID, deleted=False, title='title', category='Sport', objective='objective', language_code='lang1', community_owned=False, status=constants.ACTIVITY_STATUS_PUBLIC, exploration_model_last_updated=datetime.datetime.utcnow())
        exp_summary_2.update_timestamps()
        exp_summary_3 = self.create_model(exp_models.ExpSummaryModel, id=self.EXP_3_ID, deleted=False, title='title', category='Architecture', objective='objective', language_code='lang1', community_owned=False, status=constants.ACTIVITY_STATUS_PUBLIC, exploration_model_last_updated=datetime.datetime.utcnow())
        exp_summary_3.update_timestamps()
        self.put_multi([exp_summary_1, exp_summary_2, exp_summary_3])
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='SUCCESS: 3')])
        exp_recommendations_model_1 = recommendations_models.ExplorationRecommendationsModel.get(self.EXP_1_ID)
        assert exp_recommendations_model_1 is not None
        self.assertEqual(exp_recommendations_model_1.recommended_exploration_ids, [self.EXP_3_ID, self.EXP_2_ID])
        exp_recommendations_model_2 = recommendations_models.ExplorationRecommendationsModel.get(self.EXP_2_ID)
        assert exp_recommendations_model_2 is not None
        self.assertEqual(exp_recommendations_model_2.recommended_exploration_ids, [self.EXP_1_ID, self.EXP_3_ID])
        exp_recommendations_model_3 = recommendations_models.ExplorationRecommendationsModel.get(self.EXP_3_ID)
        assert exp_recommendations_model_3 is not None
        self.assertEqual(exp_recommendations_model_3.recommended_exploration_ids, [self.EXP_1_ID, self.EXP_2_ID])