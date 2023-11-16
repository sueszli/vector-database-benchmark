from flask_appbuilder.api import expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import has_access
from flask_babel import lazy_gettext as _
from superset.constants import MODEL_VIEW_RW_METHOD_PERMISSION_MAP, RouteMethod
from superset.models import core as models
from superset.superset_typing import FlaskResponse
from superset.views.base import DeleteMixin, SupersetModelView

class CssTemplateModelView(SupersetModelView, DeleteMixin):
    datamodel = SQLAInterface(models.CssTemplate)
    include_route_methods = RouteMethod.CRUD_SET
    class_permission_name = 'CssTemplate'
    method_permission_name = MODEL_VIEW_RW_METHOD_PERMISSION_MAP
    list_title = _('CSS Templates')
    show_title = _('Show CSS Template')
    add_title = _('Add CSS Template')
    edit_title = _('Edit CSS Template')
    list_columns = ['template_name']
    edit_columns = ['template_name', 'css']
    add_columns = edit_columns
    label_columns = {'template_name': _('Template Name')}

    @expose('/list/')
    @has_access
    def list(self) -> FlaskResponse:
        if False:
            return 10
        return super().render_app_template()

class CssTemplateAsyncModelView(CssTemplateModelView):
    include_route_methods = {RouteMethod.API_READ}
    class_permission_name = 'CssTemplate'
    method_permission_name = MODEL_VIEW_RW_METHOD_PERMISSION_MAP
    list_columns = ['template_name', 'css']