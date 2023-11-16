from datetime import datetime
from typing import Optional
import pytest
from tests.unit_tests.db_engine_specs.utils import assert_convert_dttm
from tests.unit_tests.fixtures.common import dttm

@pytest.mark.parametrize('target_type,expected_result', [('Date', "'2019-01-02'"), ('TimeStamp', "'2019-01-02 03:04:05'"), ('UnknownType', None)])
def test_convert_dttm(target_type: str, expected_result: Optional[str], dttm: datetime) -> None:
    if False:
        i = 10
        return i + 15
    from superset.db_engine_specs.cockroachdb import CockroachDbEngineSpec as spec
    assert_convert_dttm(spec, target_type, expected_result, dttm)