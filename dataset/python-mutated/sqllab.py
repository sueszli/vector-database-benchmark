import contextlib
import simplejson as json
from flask import request
from flask_appbuilder import permission_name
from flask_appbuilder.api import expose
from flask_appbuilder.security.decorators import has_access
from superset import event_logger
from superset.constants import MODEL_API_RW_METHOD_PERMISSION_MAP
from superset.superset_typing import FlaskResponse
from .base import BaseSupersetView

class SqllabView(BaseSupersetView):
    route_base = '/sqllab'
    class_permission_name = 'SQLLab'
    method_permission_name = MODEL_API_RW_METHOD_PERMISSION_MAP

    @expose('/', methods=['GET', 'POST'])
    @has_access
    @permission_name('read')
    @event_logger.log_this
    def root(self) -> FlaskResponse:
        if False:
            i = 10
            return i + 15
        payload = {}
        if (form_data := request.form.get('form_data')):
            with contextlib.suppress(json.JSONDecodeError):
                payload['requested_query'] = json.loads(form_data)
        return self.render_app_template(payload)

    @expose('/history/', methods=('GET',))
    @has_access
    @permission_name('read')
    @event_logger.log_this
    def history(self) -> FlaskResponse:
        if False:
            for i in range(10):
                print('nop')
        return self.render_app_template()