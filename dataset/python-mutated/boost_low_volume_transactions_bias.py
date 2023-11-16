from typing import List
from sentry.dynamic_sampling.rules.biases.base import Bias
from sentry.dynamic_sampling.rules.utils import RESERVED_IDS, PolymorphicRule, Rule, RuleType
from sentry.dynamic_sampling.tasks.helpers.boost_low_volume_transactions import get_transactions_resampling_rates
from sentry.models.project import Project

class BoostLowVolumeTransactionsBias(Bias):

    def generate_rules(self, project: Project, base_sample_rate: float) -> List[PolymorphicRule]:
        if False:
            i = 10
            return i + 15
        proj_id = project.id
        org_id = project.organization.id
        (transaction_map, base_implicit_rate) = get_transactions_resampling_rates(org_id=org_id, proj_id=proj_id, default_rate=base_sample_rate)
        ret_val: List[Rule] = []
        if len(transaction_map) == 0:
            return ret_val
        if base_sample_rate == 0:
            return ret_val
        if base_implicit_rate == 0.0:
            base_implicit_rate = 1.0
        implicit_rate = base_implicit_rate / base_sample_rate
        idx = 0
        for (name, base_transaction_rate) in transaction_map.items():
            transaction_rate = base_transaction_rate / base_implicit_rate
            if transaction_rate != 1.0:
                ret_val.append({'samplingValue': {'type': 'factor', 'value': transaction_rate}, 'type': 'trace', 'condition': {'op': 'or', 'inner': [{'op': 'eq', 'name': 'trace.transaction', 'value': [name], 'options': {'ignoreCase': True}}]}, 'id': RESERVED_IDS[RuleType.BOOST_LOW_VOLUME_TRANSACTIONS_RULE] + idx})
                idx += 1
        if implicit_rate != 1.0:
            ret_val.append({'samplingValue': {'type': 'factor', 'value': implicit_rate}, 'type': 'trace', 'condition': {'op': 'and', 'inner': []}, 'id': RESERVED_IDS[RuleType.BOOST_LOW_VOLUME_TRANSACTIONS_RULE] + idx})
        return ret_val