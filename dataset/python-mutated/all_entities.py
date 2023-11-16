import logging
from flask_appbuilder import expose
from flask_appbuilder.hooks import before_request
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import has_access
from jinja2.sandbox import SandboxedEnvironment
from werkzeug.exceptions import NotFound
from superset import is_feature_enabled
from superset.jinja_context import ExtraCache
from superset.superset_typing import FlaskResponse
from superset.tags.models import Tag
from superset.views.base import SupersetModelView
from .base import BaseSupersetView
logger = logging.getLogger(__name__)

def process_template(content: str) -> str:
    if False:
        i = 10
        return i + 15
    env = SandboxedEnvironment()
    template = env.from_string(content)
    context = {'current_user_id': ExtraCache.current_user_id, 'current_username': ExtraCache.current_username}
    return template.render(context)

class TaggedObjectsModelView(SupersetModelView):
    route_base = '/superset/all_entities'
    datamodel = SQLAInterface(Tag)
    class_permission_name = 'Tags'

    @has_access
    @expose('/')
    def list(self) -> FlaskResponse:
        if False:
            while True:
                i = 10
        if not is_feature_enabled('TAGGING_SYSTEM'):
            return super().list()
        return super().render_app_template()

class TaggedObjectView(BaseSupersetView):

    @staticmethod
    def is_enabled() -> bool:
        if False:
            i = 10
            return i + 15
        return is_feature_enabled('TAGGING_SYSTEM')

    @before_request
    def ensure_enabled(self) -> None:
        if False:
            return 10
        if not self.is_enabled():
            raise NotFound()