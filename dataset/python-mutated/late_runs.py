"""
The MarkLateRuns service. Responsible for putting flow runs in a Late state if they are not started on time.
The threshold for a late run can be configured by changing `PREFECT_API_SERVICES_LATE_RUNS_AFTER_SECONDS`.
"""
import asyncio
import datetime
import pendulum
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
import prefect.server.models as models
from prefect.server.database.dependencies import inject_db
from prefect.server.database.interface import PrefectDBInterface
from prefect.server.schemas import states
from prefect.server.services.loop_service import LoopService
from prefect.settings import PREFECT_API_SERVICES_LATE_RUNS_AFTER_SECONDS, PREFECT_API_SERVICES_LATE_RUNS_LOOP_SECONDS

class MarkLateRuns(LoopService):
    """
    A simple loop service responsible for identifying flow runs that are "late".

    A flow run is defined as "late" if has not scheduled within a certain amount
    of time after its scheduled start time. The exact amount is configurable in
    Prefect REST API Settings.
    """

    def __init__(self, loop_seconds: float=None, **kwargs):
        if False:
            return 10
        super().__init__(loop_seconds=loop_seconds or PREFECT_API_SERVICES_LATE_RUNS_LOOP_SECONDS.value(), **kwargs)
        self.mark_late_after: datetime.timedelta = PREFECT_API_SERVICES_LATE_RUNS_AFTER_SECONDS.value()
        self.batch_size = 400

    @inject_db
    async def run_once(self, db: PrefectDBInterface):
        """
        Mark flow runs as late by:

        - Querying for flow runs in a scheduled state that are Scheduled to start in the past
        - For any runs past the "late" threshold, setting the flow run state to a new `Late` state
        """
        scheduled_to_start_before = pendulum.now('UTC').subtract(seconds=self.mark_late_after.total_seconds())
        while True:
            async with db.session_context(begin_transaction=True) as session:
                query = self._get_select_late_flow_runs_query(scheduled_to_start_before=scheduled_to_start_before, db=db)
                result = await session.execute(query)
                runs = result.all()
                for run in runs:
                    await self._mark_flow_run_as_late(session=session, flow_run=run)
                if len(runs) < self.batch_size:
                    break
        self.logger.info('Finished monitoring for late runs.')

    @inject_db
    def _get_select_late_flow_runs_query(self, scheduled_to_start_before: datetime.datetime, db: PrefectDBInterface):
        if False:
            while True:
                i = 10
        '\n        Returns a sqlalchemy query for late flow runs.\n\n        Args:\n            scheduled_to_start_before: the maximum next scheduled start time of\n                scheduled flow runs to consider in the returned query\n        '
        query = sa.select(db.FlowRun.id, db.FlowRun.next_scheduled_start_time).where(db.FlowRun.next_scheduled_start_time <= scheduled_to_start_before, db.FlowRun.state_type == states.StateType.SCHEDULED, db.FlowRun.state_name == 'Scheduled').limit(self.batch_size)
        return query

    async def _mark_flow_run_as_late(self, session: AsyncSession, flow_run: PrefectDBInterface.FlowRun) -> None:
        """
        Mark a flow run as late.

        Pass-through method for overrides.
        """
        await models.flow_runs.set_flow_run_state(session=session, flow_run_id=flow_run.id, state=states.Late(scheduled_time=flow_run.next_scheduled_start_time), force=True)
if __name__ == '__main__':
    asyncio.run(MarkLateRuns().start())