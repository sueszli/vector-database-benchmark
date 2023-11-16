from flask_appbuilder import permission_name
from flask_appbuilder.api import expose
from flask_appbuilder.security.decorators import has_access
from superset import event_logger
from superset.superset_typing import FlaskResponse
from .base import BaseSupersetView

class ExploreView(BaseSupersetView):
    route_base = '/explore'
    class_permission_name = 'Explore'

    @expose('/')
    @has_access
    @permission_name('read')
    @event_logger.log_this
    def root(self) -> FlaskResponse:
        if False:
            return 10
        return super().render_app_template()

class ExplorePermalinkView(BaseSupersetView):
    route_base = '/superset'
    class_permission_name = 'Explore'

    @expose('/explore/p/<key>/')
    @has_access
    @permission_name('read')
    @event_logger.log_this
    def permalink(self, key: str) -> FlaskResponse:
        if False:
            return 10
        return super().render_app_template()