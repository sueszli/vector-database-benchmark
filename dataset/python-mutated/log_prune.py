import logging
from datetime import datetime, timedelta
from superset.commands.base import BaseCommand
from superset.daos.exceptions import DAODeleteFailedError
from superset.daos.report import ReportScheduleDAO
from superset.reports.commands.exceptions import ReportSchedulePruneLogError
from superset.reports.models import ReportSchedule
from superset.utils.celery import session_scope
logger = logging.getLogger(__name__)

class AsyncPruneReportScheduleLogCommand(BaseCommand):
    """
    Prunes logs from all report schedules
    """

    def __init__(self, worker_context: bool=True):
        if False:
            i = 10
            return i + 15
        self._worker_context = worker_context

    def run(self) -> None:
        if False:
            while True:
                i = 10
        with session_scope(nullpool=True) as session:
            self.validate()
            prune_errors = []
            for report_schedule in session.query(ReportSchedule).all():
                if report_schedule.log_retention is not None:
                    from_date = datetime.utcnow() - timedelta(days=report_schedule.log_retention)
                    try:
                        row_count = ReportScheduleDAO.bulk_delete_logs(report_schedule, from_date, session=session, commit=False)
                        logger.info('Deleted %s logs for report schedule id: %s', str(row_count), str(report_schedule.id))
                    except DAODeleteFailedError as ex:
                        prune_errors.append(str(ex))
            if prune_errors:
                raise ReportSchedulePruneLogError(';'.join(prune_errors))

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass