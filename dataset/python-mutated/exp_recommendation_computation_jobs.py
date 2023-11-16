"""Jobs that are run by CRON scheduler."""
from __future__ import annotations
from core.domain import exp_domain
from core.domain import exp_fetchers
from core.domain import recommendations_services
from core.jobs import base_jobs
from core.jobs.io import ndb_io
from core.jobs.transforms import job_result_transforms
from core.jobs.types import job_run_result
from core.platform import models
import apache_beam as beam
from typing import Dict, Final, Iterable, List, Tuple, Union
MYPY = False
if MYPY:
    from mypy_imports import datastore_services
    from mypy_imports import exp_models
    from mypy_imports import recommendations_models
(exp_models, recommendations_models) = models.Registry.import_models([models.Names.EXPLORATION, models.Names.RECOMMENDATIONS])
datastore_services = models.Registry.import_datastore_services()
MAX_RECOMMENDATIONS: Final = 10
SIMILARITY_SCORE_THRESHOLD: Final = 3.0

class ComputeExplorationRecommendationsJob(base_jobs.JobBase):
    """Job that indexes the explorations in Elastic Search."""

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            while True:
                i = 10
        "Returns a PCollection of 'SUCCESS' or 'FAILURE' results from\n        the Elastic Search.\n\n        Returns:\n            PCollection. A PCollection of 'SUCCESS' or 'FAILURE' results from\n            the Elastic Search.\n        "
        exp_summary_objects = self.pipeline | 'Get all non-deleted models' >> ndb_io.GetModels(exp_models.ExpSummaryModel.get_all()) | 'Convert ExpSummaryModels to domain objects' >> beam.Map(exp_fetchers.get_exploration_summary_from_model)
        exp_summary_iter = beam.pvalue.AsIter(exp_summary_objects)
        exp_recommendations_models = exp_summary_objects | 'Compute similarity' >> beam.ParDo(ComputeSimilarity(), exp_summary_iter) | 'Group similarities per exploration ID' >> beam.GroupByKey() | 'Sort and slice similarities' >> beam.MapTuple(lambda exp_id, similarities: (exp_id, self._sort_and_slice_similarities(similarities))) | 'Create recommendation models' >> beam.MapTuple(self._create_recommendation)
        unused_put_result = exp_recommendations_models | 'Put models into the datastore' >> ndb_io.PutModels()
        return exp_recommendations_models | 'Create job run result' >> job_result_transforms.CountObjectsToJobRunResult()

    @staticmethod
    def _sort_and_slice_similarities(similarities: Iterable[Dict[str, Union[str, float]]]) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Sorts similarities of explorations and slices them to\n        a maximum length.\n\n        Args:\n            similarities:iterable(). Iterable of dictionaries. The structure of\n                the dictionaries is:\n                    exp_id: str. The ID of the similar exploration.\n                    similarity_score: float. The similarity score for\n                        the exploration.\n\n        Returns:\n            list(str). List of exploration IDs, sorted by the similarity.\n        '
        sorted_similarities = sorted(similarities, reverse=True, key=lambda x: x['similarity_score'])
        return [str(item['exp_id']) for item in sorted_similarities][:MAX_RECOMMENDATIONS]

    @staticmethod
    def _create_recommendation(exp_id: str, recommended_exp_ids: Iterable[str]) -> recommendations_models.ExplorationRecommendationsModel:
        if False:
            print('Hello World!')
        'Creates exploration recommendation model.\n\n        Args:\n            exp_id: str. The exploration ID for which the recommendation is\n                created.\n            recommended_exp_ids: list(str). The list of recommended\n                exploration IDs.\n\n        Returns:\n            ExplorationRecommendationsModel. The created model.\n        '
        with datastore_services.get_ndb_context():
            exp_recommendation_model = recommendations_models.ExplorationRecommendationsModel(id=exp_id, recommended_exploration_ids=recommended_exp_ids)
        exp_recommendation_model.update_timestamps()
        return exp_recommendation_model

class ComputeSimilarity(beam.DoFn):
    """DoFn to compute similarities between exploration."""

    def process(self, ref_exp_summary: exp_domain.ExplorationSummary, compared_exp_summaries: Iterable[exp_domain.ExplorationSummary]) -> Iterable[Tuple[str, Dict[str, Union[str, float]]]]:
        if False:
            for i in range(10):
                print('nop')
        'Compute similarities between exploraitons.\n\n        Args:\n            ref_exp_summary: ExplorationSummary. Reference exploration\n                summary. We are trying to find explorations similar to this\n                reference summary.\n            compared_exp_summaries: list(ExplorationSummary). List of other\n                explorations summaries against which we compare the reference\n                summary.\n\n        Yields:\n            (str, dict(str, str|float)). Tuple, the first element is\n            the exploration ID of the reference exploration summary.\n            The second is a dictionary. The structure of the dictionary is:\n                exp_id: str. The ID of the similar exploration.\n                similarity_score: float. The similarity score for\n                    the exploration.\n        '
        with datastore_services.get_ndb_context():
            for compared_exp_summary in compared_exp_summaries:
                if compared_exp_summary.id == ref_exp_summary.id:
                    continue
                similarity_score = recommendations_services.get_item_similarity(ref_exp_summary, compared_exp_summary)
                if similarity_score >= SIMILARITY_SCORE_THRESHOLD:
                    yield (ref_exp_summary.id, {'similarity_score': similarity_score, 'exp_id': compared_exp_summary.id})