"""This module defines dep for making sure DagRun not a backfill."""
from __future__ import annotations
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.utils.session import provide_session
from airflow.utils.types import DagRunType

class DagRunNotBackfillDep(BaseTIDep):
    """Dep for valid DagRun run_id to schedule from scheduler."""
    NAME = 'DagRun is not backfill job'
    IGNORABLE = True

    @provide_session
    def _get_dep_statuses(self, ti, session, dep_context=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine if the DagRun is valid for scheduling from scheduler.\n\n        :param ti: the task instance to get the dependency status for\n        :param session: database session\n        :param dep_context: the context for which this dependency should be evaluated for\n        :return: True if DagRun is valid for scheduling from scheduler.\n        '
        dagrun = ti.get_dagrun(session)
        if dagrun.run_type == DagRunType.BACKFILL_JOB:
            yield self._failing_status(reason=f"Task's DagRun run_type is {dagrun.run_type} and cannot be run by the scheduler")