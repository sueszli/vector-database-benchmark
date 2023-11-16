from datetime import datetime
from typing import Optional
import pytest
from tests.unit_tests.db_engine_specs.utils import assert_convert_dttm
from tests.unit_tests.fixtures.common import dttm

@pytest.mark.parametrize('target_type,expected_result', [('Date', "TO_DATE('2019-01-02', 'YYYY-MM-DD')"), ('TimeStamp', 'TO_TIMESTAMP(\'2019-01-02T03:04:05.678900\', \'YYYY-MM-DD"T"HH24:MI:SS.ff6\')'), ('UnknownType', None)])
def test_convert_dttm(target_type: str, expected_result: Optional[str], dttm: datetime) -> None:
    if False:
        print('Hello World!')
    from superset.db_engine_specs.hana import HanaEngineSpec as spec
    assert_convert_dttm(spec, target_type, expected_result, dttm)