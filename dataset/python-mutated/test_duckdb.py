from datetime import datetime
from typing import Optional
import pytest
from tests.unit_tests.db_engine_specs.utils import assert_convert_dttm
from tests.unit_tests.fixtures.common import dttm

@pytest.mark.parametrize('target_type,expected_result', [('Text', "'2019-01-02 03:04:05.678900'"), ('DateTime', "'2019-01-02 03:04:05.678900'"), ('UnknownType', None)])
def test_convert_dttm(target_type: str, expected_result: Optional[str], dttm: datetime) -> None:
    if False:
        while True:
            i = 10
    from superset.db_engine_specs.duckdb import DuckDBEngineSpec as spec
    assert_convert_dttm(spec, target_type, expected_result, dttm)