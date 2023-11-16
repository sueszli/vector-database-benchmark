from __future__ import annotations
import logging
from typing import Any, TYPE_CHECKING
import simplejson as json
import superset.utils.core as utils
from superset.sqllab.command_status import SqlJsonExecutionStatus
from superset.sqllab.utils import apply_display_max_row_configuration_if_require
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from superset.models.sql_lab import Query
    from superset.sqllab.sql_json_executer import SqlResults
    from superset.sqllab.sqllab_execution_context import SqlJsonExecutionContext

class ExecutionContextConvertor:
    _max_row_in_display_configuration: int
    _exc_status: SqlJsonExecutionStatus
    payload: dict[str, Any]

    def set_max_row_in_display(self, value: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._max_row_in_display_configuration = value

    def set_payload(self, execution_context: SqlJsonExecutionContext, execution_status: SqlJsonExecutionStatus) -> None:
        if False:
            i = 10
            return i + 15
        self._exc_status = execution_status
        if execution_status == SqlJsonExecutionStatus.HAS_RESULTS:
            self.payload = execution_context.get_execution_result() or {}
        else:
            self.payload = execution_context.query.to_dict()

    def serialize_payload(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self._exc_status == SqlJsonExecutionStatus.HAS_RESULTS:
            return json.dumps(apply_display_max_row_configuration_if_require(self.payload, self._max_row_in_display_configuration), default=utils.pessimistic_json_iso_dttm_ser, ignore_nan=True, encoding=None)
        return json.dumps({'query': self.payload}, default=utils.json_int_dttm_ser, ignore_nan=True)