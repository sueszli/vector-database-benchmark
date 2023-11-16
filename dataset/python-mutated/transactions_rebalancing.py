from dataclasses import dataclass
from typing import List, Optional, Tuple
from sentry.dynamic_sampling.models.base import Model, ModelInput, ModelType
from sentry.dynamic_sampling.models.common import RebalancedItem, sum_classes_counts
from sentry.dynamic_sampling.models.full_rebalancing import FullRebalancingInput

@dataclass
class TransactionsRebalancingInput(ModelInput):
    classes: List[RebalancedItem]
    sample_rate: float
    total_num_classes: Optional[int]
    total: Optional[float]
    intensity: float

    def validate(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return 0.0 <= self.sample_rate <= 1.0 and 0.0 <= self.intensity <= 1.0 and (len(self.classes) > 0)

class TransactionsRebalancingModel(Model[TransactionsRebalancingInput, Tuple[List[RebalancedItem], float]]):

    def _run(self, model_input: TransactionsRebalancingInput) -> Tuple[List[RebalancedItem], float]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Adjusts sampling rates to bring the number of samples kept in each class as close to\n        the same value as possible while maintaining the overall sampling rate.\n\n        The algorithm adjusts the explicitly given classes individually to bring them to\n        the ideal sample rate and then adjusts the global sample rate for all the remaining classes.\n\n        :param model_input.classes: a list of class id, num_samples in class\n        :param model_input.sample_rate: global rate of sampling desired\n        :param model_input.total_num_classes: total number of classes (including the explicitly specified in classes)\n        :param model_input.intensity: the adjustment strength 0: no adjustment, 1: try to bring everything to mean\n        :param model_input.total: total number of samples in all classes (including the explicitly specified classes)\n\n        :return: a list of items with calculated sample_rates and a rate for all other (unspecified) classes.\n        '
        classes = model_input.classes
        sample_rate = model_input.sample_rate
        total_num_classes = model_input.total_num_classes
        total = model_input.total
        intensity = model_input.intensity
        classes = sorted(classes, key=lambda x: (x.count, x.id), reverse=True)
        total_explicit = sum_classes_counts(classes)
        if total is None:
            total = total_explicit
        if total_num_classes is None:
            total_num_classes = len(classes)
        total_implicit = total - total_explicit
        num_explicit_classes = len(classes)
        num_implicit_classes = total_num_classes - num_explicit_classes
        total_budget = total * sample_rate
        budget_per_class = total_budget / total_num_classes
        implicit_budget = budget_per_class * num_implicit_classes
        explicit_budget = budget_per_class * num_explicit_classes
        from sentry.dynamic_sampling.models.factory import model_factory
        full_rebalancing = model_factory(ModelType.FULL_REBALANCING)
        if num_explicit_classes == total_num_classes:
            (explicit_rates, _used) = full_rebalancing.run(FullRebalancingInput(classes=classes, sample_rate=sample_rate, intensity=intensity))
            implicit_rate = sample_rate
        elif total_implicit < implicit_budget:
            implicit_rate = 1
            explicit_budget = total_budget - total_implicit
            explicit_rate = explicit_budget / total_explicit
            (explicit_rates, _used) = full_rebalancing.run(FullRebalancingInput(classes=classes, sample_rate=explicit_rate, intensity=intensity))
        elif total_explicit < explicit_budget:
            explicit_rates = [RebalancedItem(id=element.id, count=element.count, new_sample_rate=1.0) for element in classes]
            implicit_budget = total_budget - total_explicit
            implicit_rate = implicit_budget / total_implicit
        else:
            minimum_explicit_budget = total_budget - total_implicit
            explicit_rate = explicit_budget / total_explicit
            (explicit_rates, used) = full_rebalancing.run(FullRebalancingInput(classes=classes, sample_rate=explicit_rate, intensity=intensity, min_budget=minimum_explicit_budget))
            implicit_budget = total_budget - used
            implicit_rate = implicit_budget / total_implicit
        return (explicit_rates, implicit_rate)