from datetime import datetime
from typing import Optional
import pytest
from tests.unit_tests.db_engine_specs.utils import assert_convert_dttm
from tests.unit_tests.fixtures.common import dttm

def test_epoch_to_dttm() -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    DB Eng Specs (crate): Test epoch to dttm\n    '
    from superset.db_engine_specs.crate import CrateEngineSpec
    assert CrateEngineSpec.epoch_to_dttm() == '{col} * 1000'

def test_epoch_ms_to_dttm() -> None:
    if False:
        print('Hello World!')
    '\n    DB Eng Specs (crate): Test epoch ms to dttm\n    '
    from superset.db_engine_specs.crate import CrateEngineSpec
    assert CrateEngineSpec.epoch_ms_to_dttm() == '{col}'

def test_alter_new_orm_column() -> None:
    if False:
        print('Hello World!')
    '\n    DB Eng Specs (crate): Test alter orm column\n    '
    from superset.connectors.sqla.models import SqlaTable, TableColumn
    from superset.db_engine_specs.crate import CrateEngineSpec
    from superset.models.core import Database
    database = Database(database_name='crate', sqlalchemy_uri='crate://db')
    tbl = SqlaTable(table_name='tbl', database=database)
    col = TableColumn(column_name='ts', type='TIMESTAMP', table=tbl)
    CrateEngineSpec.alter_new_orm_column(col)
    assert col.python_date_format == 'epoch_ms'

@pytest.mark.parametrize('target_type,expected_result', [('TimeStamp', '1546398245678.9'), ('UnknownType', None)])
def test_convert_dttm(target_type: str, expected_result: Optional[str], dttm: datetime) -> None:
    if False:
        print('Hello World!')
    from superset.db_engine_specs.crate import CrateEngineSpec as spec
    assert_convert_dttm(spec, target_type, expected_result, dttm)