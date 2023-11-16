from __future__ import annotations
import pytest
from airflow.utils.trigger_rule import TriggerRule

class TestTriggerRule:

    def test_valid_trigger_rules(self):
        if False:
            for i in range(10):
                print('nop')
        assert TriggerRule.is_valid(TriggerRule.ALL_SUCCESS)
        assert TriggerRule.is_valid(TriggerRule.ALL_FAILED)
        assert TriggerRule.is_valid(TriggerRule.ALL_DONE)
        assert TriggerRule.is_valid(TriggerRule.ALL_SKIPPED)
        assert TriggerRule.is_valid(TriggerRule.ONE_SUCCESS)
        assert TriggerRule.is_valid(TriggerRule.ONE_FAILED)
        assert TriggerRule.is_valid(TriggerRule.ONE_DONE)
        assert TriggerRule.is_valid(TriggerRule.NONE_FAILED)
        assert TriggerRule.is_valid(TriggerRule.NONE_FAILED_OR_SKIPPED)
        assert TriggerRule.is_valid(TriggerRule.NONE_SKIPPED)
        assert TriggerRule.is_valid(TriggerRule.DUMMY)
        assert TriggerRule.is_valid(TriggerRule.ALWAYS)
        assert TriggerRule.is_valid(TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
        assert TriggerRule.is_valid(TriggerRule.ALL_DONE_SETUP_SUCCESS)
        assert len(TriggerRule.all_triggers()) == 14
        with pytest.raises(ValueError):
            TriggerRule('NOT_EXIST_TRIGGER_RULE')