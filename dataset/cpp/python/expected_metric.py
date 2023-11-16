from dataclasses import dataclass
from typing import Union

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ExpectedMetric:
    # Output feature name.
    output_feature_name: str

    # Metric name.
    metric_name: str

    # Expected metric value.
    expected_value: Union[int, float]

    # The percentage change that would trigger a notification/failure.
    tolerance_percentage: float
