from typing import Any, Optional, Union
import simplejson as json
from flask import g
from superset.charts.commands.exceptions import ChartInvalidError, WarmUpCacheChartNotFoundError
from superset.charts.data.commands.get_data_command import ChartDataCommand
from superset.commands.base import BaseCommand
from superset.extensions import db
from superset.models.slice import Slice
from superset.utils.core import error_msg_from_exception
from superset.views.utils import get_dashboard_extra_filters, get_form_data, get_viz
from superset.viz import viz_types

class ChartWarmUpCacheCommand(BaseCommand):

    def __init__(self, chart_or_id: Union[int, Slice], dashboard_id: Optional[int], extra_filters: Optional[str]):
        if False:
            return 10
        self._chart_or_id = chart_or_id
        self._dashboard_id = dashboard_id
        self._extra_filters = extra_filters

    def run(self) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        self.validate()
        chart: Slice = self._chart_or_id
        try:
            form_data = get_form_data(chart.id, use_slice_data=True)[0]
            if form_data.get('viz_type') in viz_types:
                if not chart.datasource:
                    raise ChartInvalidError("Chart's datasource does not exist")
                if self._dashboard_id:
                    form_data['extra_filters'] = json.loads(self._extra_filters) if self._extra_filters else get_dashboard_extra_filters(chart.id, self._dashboard_id)
                g.form_data = form_data
                payload = get_viz(datasource_type=chart.datasource.type, datasource_id=chart.datasource.id, form_data=form_data, force=True).get_payload()
                delattr(g, 'form_data')
                error = payload['errors'] or None
                status = payload['status']
            else:
                query_context = chart.get_query_context()
                if not query_context:
                    raise ChartInvalidError("Chart's query context does not exist")
                query_context.force = True
                command = ChartDataCommand(query_context)
                command.validate()
                payload = command.run()
                for query in payload['queries']:
                    error = query['error']
                    status = query['status']
                    if error is not None:
                        break
        except Exception as ex:
            error = error_msg_from_exception(ex)
            status = None
        return {'chart_id': chart.id, 'viz_error': error, 'viz_status': status}

    def validate(self) -> None:
        if False:
            return 10
        if isinstance(self._chart_or_id, Slice):
            return
        chart = db.session.query(Slice).filter_by(id=self._chart_or_id).scalar()
        if not chart:
            raise WarmUpCacheChartNotFoundError()
        self._chart_or_id = chart