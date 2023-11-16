from __future__ import annotations
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep

class MappedTaskIsExpanded(BaseTIDep):
    """Checks that a mapped task has been expanded before its TaskInstance can run."""
    NAME = 'Task has been mapped'
    IGNORABLE = False
    IS_TASK_DEP = False

    def _get_dep_statuses(self, ti, session, dep_context):
        if False:
            while True:
                i = 10
        if dep_context.ignore_unmapped_tasks:
            return
        if ti.map_index == -1:
            yield self._failing_status(reason='The task has yet to be mapped!')
            return
        yield self._passing_status(reason='The task has been mapped')