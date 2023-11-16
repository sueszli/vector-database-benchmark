from __future__ import annotations
from unittest.mock import Mock
import pytest
from airflow.models.dagrun import DagRun
from airflow.ti_deps.deps.dagrun_backfill_dep import DagRunNotBackfillDep
from airflow.utils.types import DagRunType
pytestmark = pytest.mark.db_test

class TestDagrunRunningDep:

    def test_run_id_is_backfill(self):
        if False:
            print('Hello World!')
        '\n        Task instances whose run_id is a backfill dagrun run_id should fail this dep.\n        '
        dagrun = DagRun()
        dagrun.run_id = 'anything'
        dagrun.run_type = DagRunType.BACKFILL_JOB
        ti = Mock(get_dagrun=Mock(return_value=dagrun))
        assert not DagRunNotBackfillDep().is_met(ti=ti)

    def test_run_id_is_not_backfill(self):
        if False:
            return 10
        '\n        Task instances whose run_id is not a backfill run_id should pass this dep.\n        '
        dagrun = DagRun()
        dagrun.run_type = 'custom_type'
        ti = Mock(get_dagrun=Mock(return_value=dagrun))
        assert DagRunNotBackfillDep().is_met(ti=ti)
        dagrun = DagRun()
        dagrun.run_id = None
        ti = Mock(get_dagrun=Mock(return_value=dagrun))
        assert DagRunNotBackfillDep().is_met(ti=ti)