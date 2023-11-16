"""Track recorder run history."""
from __future__ import annotations
import bisect
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.orm.session import Session
import homeassistant.util.dt as dt_util
from ..db_schema import RecorderRuns
from ..models import process_timestamp

def _find_recorder_run_for_start_time(run_history: _RecorderRunsHistory, start: datetime) -> RecorderRuns | None:
    if False:
        return 10
    'Find the recorder run for a start time in _RecorderRunsHistory.'
    run_timestamps = run_history.run_timestamps
    runs_by_timestamp = run_history.runs_by_timestamp
    if (idx := bisect.bisect_left(run_timestamps, start.timestamp())):
        return runs_by_timestamp[run_timestamps[idx - 1]]
    return None

@dataclass(frozen=True)
class _RecorderRunsHistory:
    """Bisectable history of RecorderRuns."""
    run_timestamps: list[int]
    runs_by_timestamp: dict[int, RecorderRuns]

class RecorderRunsManager:
    """Track recorder run history."""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        'Track recorder run history.'
        self._recording_start = dt_util.utcnow()
        self._current_run_info: RecorderRuns | None = None
        self._run_history = _RecorderRunsHistory([], {})

    @property
    def recording_start(self) -> datetime:
        if False:
            return 10
        'Return the time the recorder started recording states.'
        return self._recording_start

    @property
    def first(self) -> RecorderRuns:
        if False:
            for i in range(10):
                print('nop')
        'Get the first run.'
        if (runs_by_timestamp := self._run_history.runs_by_timestamp):
            return next(iter(runs_by_timestamp.values()))
        return self.current

    @property
    def current(self) -> RecorderRuns:
        if False:
            while True:
                i = 10
        'Get the current run.'
        return self._current_run_info or RecorderRuns(start=self.recording_start, created=dt_util.utcnow())

    @property
    def active(self) -> bool:
        if False:
            return 10
        'Return if a run is active.'
        return self._current_run_info is not None

    def get(self, start: datetime) -> RecorderRuns | None:
        if False:
            i = 10
            return i + 15
        'Return the recorder run that started before or at start.\n\n        If the first run started after the start, return None\n        '
        if start >= self.recording_start:
            return self.current
        return _find_recorder_run_for_start_time(self._run_history, start)

    def start(self, session: Session) -> None:
        if False:
            i = 10
            return i + 15
        'Start a new run.\n\n        Must run in the recorder thread.\n        '
        self._current_run_info = RecorderRuns(start=self.recording_start, created=dt_util.utcnow())
        session.add(self._current_run_info)
        session.flush()
        session.expunge(self._current_run_info)
        self.load_from_db(session)

    def reset(self) -> None:
        if False:
            return 10
        'Reset the run when the database is changed or fails.\n\n        Must run in the recorder thread.\n        '
        self._recording_start = dt_util.utcnow()
        self._current_run_info = None

    def end(self, session: Session) -> None:
        if False:
            i = 10
            return i + 15
        'End the current run.\n\n        Must run in the recorder thread.\n        '
        assert self._current_run_info is not None
        self._current_run_info.end = dt_util.utcnow()
        session.add(self._current_run_info)

    def load_from_db(self, session: Session) -> None:
        if False:
            print('Hello World!')
        'Update the run cache.\n\n        Must run in the recorder thread.\n        '
        run_timestamps: list[int] = []
        runs_by_timestamp: dict[int, RecorderRuns] = {}
        for run in session.query(RecorderRuns).order_by(RecorderRuns.start.asc()).all():
            session.expunge(run)
            if (run_dt := process_timestamp(run.start)):
                timestamp = int(run_dt.timestamp())
                run_timestamps.append(timestamp)
                runs_by_timestamp[timestamp] = run
        self._run_history = _RecorderRunsHistory(run_timestamps, runs_by_timestamp)

    def clear(self) -> None:
        if False:
            print('Hello World!')
        'Clear the current run after ending it.\n\n        Must run in the recorder thread.\n        '
        if self._current_run_info:
            self._current_run_info = None