from __future__ import annotations
from typing import Any, cast, TYPE_CHECKING
from superset.common.chart_data import ChartDataResultType
from superset.common.query_object import QueryObject
from superset.common.utils.time_range_utils import get_since_until_from_time_range
from superset.constants import NO_TIME_RANGE
from superset.superset_typing import Column
from superset.utils.core import apply_max_row_limit, DatasourceDict, DatasourceType, FilterOperator, get_xaxis_label, QueryObjectFilterClause
if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker
    from superset.connectors.base.models import BaseDatasource
    from superset.daos.datasource import DatasourceDAO

class QueryObjectFactory:
    _config: dict[str, Any]
    _datasource_dao: DatasourceDAO
    _session_maker: sessionmaker

    def __init__(self, app_configurations: dict[str, Any], _datasource_dao: DatasourceDAO, session_maker: sessionmaker):
        if False:
            while True:
                i = 10
        self._config = app_configurations
        self._datasource_dao = _datasource_dao
        self._session_maker = session_maker

    def create(self, parent_result_type: ChartDataResultType, datasource: DatasourceDict | None=None, extras: dict[str, Any] | None=None, row_limit: int | None=None, time_range: str | None=None, time_shift: str | None=None, **kwargs: Any) -> QueryObject:
        if False:
            for i in range(10):
                print('nop')
        datasource_model_instance = None
        if datasource:
            datasource_model_instance = self._convert_to_model(datasource)
        processed_extras = self._process_extras(extras)
        result_type = kwargs.setdefault('result_type', parent_result_type)
        row_limit = self._process_row_limit(row_limit, result_type)
        processed_time_range = self._process_time_range(time_range, kwargs.get('filters'), kwargs.get('columns'))
        (from_dttm, to_dttm) = get_since_until_from_time_range(processed_time_range, time_shift, processed_extras)
        kwargs['from_dttm'] = from_dttm
        kwargs['to_dttm'] = to_dttm
        return QueryObject(datasource=datasource_model_instance, extras=extras, row_limit=row_limit, time_range=time_range, time_shift=time_shift, **kwargs)

    def _convert_to_model(self, datasource: DatasourceDict) -> BaseDatasource:
        if False:
            return 10
        return self._datasource_dao.get_datasource(datasource_type=DatasourceType(datasource['type']), datasource_id=int(datasource['id']), session=self._session_maker())

    def _process_extras(self, extras: dict[str, Any] | None) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        extras = extras or {}
        return extras

    def _process_row_limit(self, row_limit: int | None, result_type: ChartDataResultType) -> int:
        if False:
            return 10
        default_row_limit = self._config['SAMPLES_ROW_LIMIT'] if result_type == ChartDataResultType.SAMPLES else self._config['ROW_LIMIT']
        return apply_max_row_limit(row_limit or default_row_limit)

    @staticmethod
    def _process_time_range(time_range: str | None, filters: list[QueryObjectFilterClause] | None=None, columns: list[Column] | None=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        if time_range is None:
            time_range = NO_TIME_RANGE
            temporal_flt = [flt for flt in filters or [] if flt.get('op') == FilterOperator.TEMPORAL_RANGE]
            if temporal_flt:
                xaxis_label = get_xaxis_label(columns or [])
                match_flt = [flt for flt in temporal_flt if flt.get('col') == xaxis_label]
                if match_flt:
                    time_range = cast(str, match_flt[0].get('val'))
                else:
                    time_range = cast(str, temporal_flt[0].get('val'))
        return time_range