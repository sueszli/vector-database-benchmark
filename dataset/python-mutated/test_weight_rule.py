from __future__ import annotations
import pytest
from airflow.utils.weight_rule import WeightRule

class TestWeightRule:

    def test_valid_weight_rules(self):
        if False:
            i = 10
            return i + 15
        assert WeightRule.is_valid(WeightRule.DOWNSTREAM)
        assert WeightRule.is_valid(WeightRule.UPSTREAM)
        assert WeightRule.is_valid(WeightRule.ABSOLUTE)
        assert len(WeightRule.all_weight_rules()) == 3
        with pytest.raises(ValueError):
            WeightRule('NOT_EXIST_WEIGHT_RULE')