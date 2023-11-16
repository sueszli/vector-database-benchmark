from dataclasses import dataclass
from typing import List, Optional, Tuple
from sentry.dynamic_sampling.models.base import Model, ModelInput
from sentry.dynamic_sampling.models.common import RebalancedItem, sum_classes_counts

@dataclass
class FullRebalancingInput(ModelInput):
    classes: List[RebalancedItem]
    sample_rate: float
    intensity: float
    min_budget: Optional[float] = None

    def validate(self) -> bool:
        if False:
            while True:
                i = 10
        return 0.0 <= self.sample_rate <= 1.0 and 0.0 <= self.intensity <= 1.0 and (len(self.classes) > 0)

class FullRebalancingModel(Model[FullRebalancingInput, Tuple[List[RebalancedItem], float]]):

    def _run(self, model_input: FullRebalancingInput) -> Tuple[List[RebalancedItem], float]:
        if False:
            while True:
                i = 10
        '\n        Tries to calculate rates that brings all counts close to the ideal count.\n\n        The intensity controls how close, 0 intensity leaves the items unchanged, 1 brings the items to the\n        ideal count ( or rate=1.0 if ideal count is too high).\n\n        :param model_input.classes: The items to be balanced :param model_input.sample_rate: The overall rate\n        necessary :param model_input.intensity: How close to the ideal should we go from our current position (0=do\n        not change, 1 go to ideal) :param model_input.min_budget: Ensure that we use at least min_budget (in order to\n        keep the overall rate)\n\n        :return: A mapping with the frequency for all items + the amount of items used (it should allways be at least\n        minimum_consumption if passed).\n        '
        classes = model_input.classes
        sample_rate = model_input.sample_rate
        intensity = model_input.intensity
        min_budget = model_input.min_budget
        total = sum_classes_counts(classes)
        num_classes = len(classes)
        if min_budget is None:
            min_budget = total * sample_rate
        assert total >= min_budget
        ideal = total * sample_rate / num_classes
        used_budget: float = 0.0
        ret_val = []
        while classes:
            element = classes.pop()
            count = element.count
            if ideal * num_classes < min_budget:
                ideal = min_budget / num_classes
            sampled = count * sample_rate
            delta = ideal - sampled
            correction = delta * intensity
            desired_count = sampled + correction
            if desired_count > count:
                new_sample_rate = 1.0
                used = count
            else:
                new_sample_rate = desired_count / count
                used = desired_count
            ret_val.append(RebalancedItem(id=element.id, count=count, new_sample_rate=new_sample_rate))
            min_budget -= used
            used_budget += used
            num_classes -= 1
        return (ret_val, used_budget)