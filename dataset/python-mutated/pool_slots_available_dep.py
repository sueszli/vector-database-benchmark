"""This module defines dep for pool slots availability."""
from __future__ import annotations
from sqlalchemy import select
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.utils.session import provide_session

class PoolSlotsAvailableDep(BaseTIDep):
    """Dep for pool slots availability."""
    NAME = 'Pool Slots Available'
    IGNORABLE = True

    @provide_session
    def _get_dep_statuses(self, ti, session, dep_context=None):
        if False:
            i = 10
            return i + 15
        '\n        Determine if the pool task instance is in has available slots.\n\n        :param ti: the task instance to get the dependency status for\n        :param session: database session\n        :param dep_context: the context for which this dependency should be evaluated for\n        :return: True if there are available slots in the pool.\n        '
        from airflow.models.pool import Pool
        pool_name = ti.pool
        pool: Pool | None = session.scalar(select(Pool).where(Pool.pool == pool_name))
        if pool is None:
            yield self._failing_status(reason=f"Tasks using non-existent pool '{pool_name}' will not be scheduled")
            return
        open_slots = pool.open_slots(session=session)
        if ti.state in pool.get_occupied_states():
            open_slots += ti.pool_slots
        if open_slots <= ti.pool_slots - 1:
            yield self._failing_status(reason=f'Not scheduling since there are {open_slots} open slots in pool {pool_name} and require {ti.pool_slots} pool slots')
        else:
            yield self._passing_status(reason=f'There are enough open slots in {pool_name} to execute the task')