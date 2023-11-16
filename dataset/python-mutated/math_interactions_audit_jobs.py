"""Audit jobs for math interactions: AlgebraicExpressionInput,
NumericExpressionInput, MathEquationInput.
"""
from __future__ import annotations
from core import feconf
from core.jobs import base_jobs
from core.jobs.io import ndb_io
from core.jobs.types import job_run_result
from core.platform import models
import apache_beam as beam
from typing import List, Tuple
MYPY = False
if MYPY:
    from core.domain import state_domain
    from mypy_imports import exp_models
(exp_models,) = models.Registry.import_models([models.Names.EXPLORATION])

class FindMathExplorationsWithRulesJob(base_jobs.JobBase):
    """Finds explorations that use at least one of the math interactions
    and accumulates the output along with the rules.

    Expected output:
    (exp_id_1, state_name_1, [rule_type_1, rule_type_2, ...])
    (exp_id_2, state_name_4, [rule_type_1, rule_type_2, ...])
    ...
    """

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            for i in range(10):
                print('nop')
        exp_models_pcoll = self.pipeline | 'Get all ExplorationModels' >> ndb_io.GetModels(exp_models.ExplorationModel.get_all())
        exp_models_filtered = exp_models_pcoll | 'Filter Math ExplorationModels' >> beam.Filter(self.contains_math_interactions)
        exp_models_with_states = exp_models_filtered | 'Mapping exp_ids with states' >> beam.FlatMap(self.flat_map_exp_with_states)
        exp_models_with_states_filtered = exp_models_with_states | 'Filtering out states without math interactions' >> beam.Filter(lambda tup: tup[2]['interaction']['id'] in feconf.MATH_INTERACTION_IDS)
        exp_models_with_states_and_rules = exp_models_with_states_filtered | 'Mapping with rule types list' >> beam.Map(self.map_with_rule_types)
        return exp_models_with_states_and_rules | 'Final output' >> beam.Map(job_run_result.JobRunResult.as_stdout)

    def contains_math_interactions(self, model: exp_models.ExplorationModel) -> bool:
        if False:
            while True:
                i = 10
        'Checks if the exploration contains any state with any of the\n        math interactions.\n\n        Args:\n            model: ExplorationModel. Model instance to be checked.\n\n        Returns:\n            bool. Whether the exploration contains math interactions.\n        '
        return any((state_dict['interaction']['id'] in feconf.MATH_INTERACTION_IDS for state_dict in model.states.values()))

    def flat_map_exp_with_states(self, model: exp_models.ExplorationModel) -> List[Tuple[str, str, state_domain.StateDict]]:
        if False:
            for i in range(10):
                print('nop')
        "Maps exploration model with it's states data.\n\n        Args:\n            model: ExplorationModel. Model instance to be mapped.\n\n        Returns:\n            List[Tuple[str, str, dict]]. List of tuples\n            (exp_id, state_name, state_dict).\n        "
        return [(model.id, state_name, state_dict) for (state_name, state_dict) in model.states.items()]

    def map_with_rule_types(self, tup: Tuple[str, str, state_domain.StateDict]) -> Tuple[str, str, List[str]]:
        if False:
            print('Hello World!')
        "Maps state tuple with it's rule types.\n\n        Args:\n            tup: Tuple[str, str, dict]. State tuple to be modified.\n\n        Returns:\n            Tuple[str, str, List[str]]. Mapped tuple\n            (exp_id, state_name, list of rules).\n        "
        answer_groups = tup[2]['interaction']['answer_groups']
        rule_types = []
        for answer_group in answer_groups:
            for rule_spec in answer_group['rule_specs']:
                rule_types.append(rule_spec['rule_type'])
        return (tup[0], tup[1], rule_types)