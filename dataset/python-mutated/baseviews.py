from datetime import date, datetime
from inspect import isclass
import json
import logging
import re
from typing import List, Optional, TYPE_CHECKING
from flask import abort, Blueprint, current_app, flash, render_template, request, session, url_for
from ._compat import as_unicode
from .actions import ActionItem
from .const import PERMISSION_PREFIX
from .forms import GeneralModelConverter
from .hooks import get_before_request_hooks, wrap_route_handler_with_hooks
from .urltools import get_filter_args, get_order_args, get_page_args, get_page_size_args, Stack
from .widgets import FormWidget, ListWidget, SearchWidget, ShowWidget
if TYPE_CHECKING:
    from flask_appbuilder.base import AppBuilder
log = logging.getLogger(__name__)

def expose(url='/', methods=('GET',)):
    if False:
        return 10
    '\n    Use this decorator to expose views on your view classes.\n\n    :param url:\n        Relative URL for the view\n    :param methods:\n        Allowed HTTP methods. By default only GET is allowed.\n    '

    def wrap(f):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(f, '_urls'):
            f._urls = []
        f._urls.append((url, methods))
        return f
    return wrap

def expose_api(name='', url='', methods=('GET',), description=''):
    if False:
        return 10

    def wrap(f):
        if False:
            while True:
                i = 10
        api_name = name or f.__name__
        api_url = url or '/api/{0}'.format(name)
        if not hasattr(f, '_urls'):
            f._urls = []
            f._extra = {}
        f._urls.append((api_url, methods))
        f._extra[api_name] = (api_url, f.__name__, description)
        return f
    return wrap

class AbstractViewApi:
    appbuilder: 'AppBuilder'
    base_permissions: Optional[List[str]]
    class_permission_name: str
    endpoint: str
    default_view: str

    def create_blueprint(self, appbuilder: 'AppBuilder', endpoint: Optional[str]=None, static_folder: Optional[str]=None):
        if False:
            while True:
                i = 10
        ...

    def get_uninit_inner_views(self):
        if False:
            return 10
        '\n        Will return a list with views that need to be initialized.\n        Normally related_views from ModelView\n        '
        ...

    def get_init_inner_views(self):
        if False:
            while True:
                i = 10
        '\n        Sets initialized inner views\n        '
        ...

class BaseView(AbstractViewApi):
    """
    All views inherit from this class.
    it's constructor will register your exposed urls on flask as a Blueprint.

    This class does not expose any urls, but provides a common base for all views.

    Extend this class if you want to expose methods for your own templates
    """
    appbuilder = None
    blueprint = None
    endpoint = None
    route_base = None
    ' Override this if you want to define your own relative url '
    template_folder = 'templates'
    ' The template folder relative location '
    static_folder = 'static'
    '  The static folder relative location '
    base_permissions = None
    "\n        List with allowed base permission.\n        Use it like this if you want to restrict your view to readonly::\n\n            class MyView(ModelView):\n                base_permissions = ['can_list','can_show']\n    "
    class_permission_name = None
    '\n        Override class permission name default fallback to self.__class__.__name__\n    '
    previous_class_permission_name = None
    '\n        If set security cleanup will remove all permissions tuples\n        with this name\n    '
    method_permission_name = None
    "\n        Override method permission names, example::\n\n            method_permissions_name = {\n                'get_list': 'read',\n                'get': 'read',\n                'put': 'write',\n                'post': 'write',\n                'delete': 'write'\n            }\n    "
    previous_method_permission_name = None
    '\n        Use same structure as method_permission_name. If set security converge\n        will replace all method permissions by the new ones\n    '
    exclude_route_methods = set()
    '\n        Does not register routes for a set of builtin ModelView functions.\n        example::\n\n            class ContactModelView(ModelView):\n                datamodel = SQLAInterface(Contact)\n                exclude_route_methods = {"delete", "edit"}\n\n    '
    include_route_methods = None
    '\n        If defined will assume a white list setup, where all endpoints are excluded\n        except those define on this attribute\n        example::\n\n            class ContactModelView(ModelView):\n                datamodel = SQLAInterface(Contact)\n                include_route_methods = {"list"}\n\n\n        The previous example will exclude all endpoints except the `list` endpoint\n    '
    default_view = 'list'
    ' the default view for this BaseView, to be used with url_for (method name) '
    extra_args = None
    ' dictionary for injecting extra arguments into template '
    limits = None
    '\n        List of limits for this view.\n\n        Use it like this if you want to restrict the rate of requests to a view:\n\n            class MyView(ModelView):\n                limits = [Limit("2 per 5 second")]\n\n        or use the decorator @limit.\n    '
    _apis = None

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Initialization of base permissions\n        based on exposed methods and actions\n\n        Initialization of extra args\n        '
        if not self.previous_class_permission_name and self.class_permission_name:
            self.previous_class_permission_name = self.__class__.__name__
        self.class_permission_name = self.class_permission_name or self.__class__.__name__
        is_collect_previous = False
        if not self.previous_method_permission_name and self.method_permission_name:
            self.previous_method_permission_name = dict()
            is_collect_previous = True
        self.method_permission_name = self.method_permission_name or dict()
        is_add_base_permissions = False
        if self.base_permissions is None:
            self.base_permissions = set()
            is_add_base_permissions = True
        if self.limits is None:
            self.limits = []
        for attr_name in dir(self):
            if self.include_route_methods is not None and attr_name not in self.include_route_methods:
                continue
            if attr_name in self.exclude_route_methods:
                continue
            if hasattr(getattr(self, attr_name), '_permission_name'):
                if is_collect_previous:
                    self.previous_method_permission_name[attr_name] = getattr(getattr(self, attr_name), '_permission_name')
                _permission_name = self.get_method_permission(attr_name)
                if is_add_base_permissions:
                    self.base_permissions.add(PERMISSION_PREFIX + _permission_name)
        self.base_permissions = list(self.base_permissions)
        if not self.extra_args:
            self.extra_args = dict()
        self._apis = dict()
        for attr_name in dir(self):
            if hasattr(getattr(self, attr_name), '_extra'):
                _extra = getattr(getattr(self, attr_name), '_extra')
                for key in _extra:
                    self._apis[key] = _extra[key]
            if hasattr(getattr(self, attr_name), '_limit'):
                self.limits.append(getattr(getattr(self, attr_name), '_limit'))

    def create_blueprint(self, appbuilder, endpoint=None, static_folder=None):
        if False:
            i = 10
            return i + 15
        '\n        Create Flask blueprint. You will generally not use it\n\n        :param appbuilder:\n           the AppBuilder object\n        :param endpoint:\n           endpoint override for this blueprint,\n           will assume class name if not provided\n        :param static_folder:\n           the relative override for static folder,\n           if omitted application will use the appbuilder static\n        '
        self.appbuilder = appbuilder
        self.endpoint = endpoint or self.__class__.__name__
        if self.route_base is None:
            self.route_base = '/' + self.__class__.__name__.lower()
        self.static_folder = static_folder
        if not static_folder:
            self.blueprint = Blueprint(self.endpoint, __name__, url_prefix=self.route_base, template_folder=self.template_folder)
        else:
            self.blueprint = Blueprint(self.endpoint, __name__, url_prefix=self.route_base, template_folder=self.template_folder, static_folder=static_folder)
        self._register_urls()
        return self.blueprint

    def _register_urls(self):
        if False:
            return 10
        before_request_hooks = get_before_request_hooks(self)
        for attr_name in dir(self):
            if self.include_route_methods is not None and attr_name not in self.include_route_methods:
                continue
            if attr_name in self.exclude_route_methods:
                log.info('Not registering route for method %s.%s', self.__class__.__name__, attr_name)
                continue
            attr = getattr(self, attr_name)
            if hasattr(attr, '_urls'):
                for (url, methods) in attr._urls:
                    log.info('Registering route %s%s %s', self.blueprint.url_prefix, url, methods)
                    route_handler = wrap_route_handler_with_hooks(attr_name, attr, before_request_hooks)
                    self.blueprint.add_url_rule(url, attr_name, route_handler, methods=methods)

    def render_template(self, template, **kwargs):
        if False:
            print('Hello World!')
        '\n        Use this method on your own endpoints, will pass the extra_args\n        to the templates.\n\n        :param template: The template relative path\n        :param kwargs: arguments to be passed to the template\n        '
        kwargs['base_template'] = self.appbuilder.base_template
        kwargs['appbuilder'] = self.appbuilder
        return render_template(template, **dict(list(kwargs.items()) + list(self.extra_args.items())))

    def _prettify_name(self, name):
        if False:
            print('Hello World!')
        "\n        Prettify pythonic variable name.\n\n        For example, 'HelloWorld' will be converted to 'Hello World'\n\n        :param name:\n            Name to prettify.\n        "
        return re.sub('(?<=.)([A-Z])', ' \\1', name)

    def _prettify_column(self, name):
        if False:
            for i in range(10):
                print('nop')
        "\n        Prettify pythonic variable name.\n\n        For example, 'hello_world' will be converted to 'Hello World'\n\n        :param name:\n            Name to prettify.\n        "
        return re.sub('[._]', ' ', name).title()

    def update_redirect(self):
        if False:
            return 10
        "\n        Call it on your own endpoint's to update the back history navigation.\n        If you bypass it, the next submit or back will go over it.\n        "
        page_history = Stack(session.get('page_history', []))
        page_history.push(request.url)
        session['page_history'] = page_history.to_json()

    def get_redirect(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the previous url.\n        '
        index_url = self.appbuilder.get_url_for_index
        page_history = Stack(session.get('page_history', []))
        if page_history.pop() is None:
            return index_url
        session['page_history'] = page_history.to_json()
        url = page_history.pop() or index_url
        return url

    @classmethod
    def get_default_url(cls, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the url for this class default endpoint\n        '
        return url_for(cls.__name__ + '.' + cls.default_view, **kwargs)

    def get_uninit_inner_views(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Will return a list with views that need to be initialized.\n        Normally related_views from ModelView\n        '
        return []

    def get_init_inner_views(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets initialized inner views\n        '

    def get_method_permission(self, method_name: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the permission name for a method\n        '
        permission = self.method_permission_name.get(method_name)
        if permission:
            return permission
        else:
            return getattr(getattr(self, method_name), '_permission_name')

class BaseFormView(BaseView):
    """
    Base class FormView's
    """
    form_template = 'appbuilder/general/model/edit.html'
    edit_widget = FormWidget
    ' Form widget to override '
    form_title = ''
    ' The form title to be displayed '
    form_columns = None
    ' The form columns to include, if empty will include all'
    form = None
    ' The WTF form to render '
    form_fieldsets = None
    ' Form field sets '
    default_view = 'this_form_get'
    ' The form view default entry endpoint '

    def _init_vars(self):
        if False:
            for i in range(10):
                print('nop')
        self.form_columns = self.form_columns or []
        self.form_fieldsets = self.form_fieldsets or []
        list_cols = [field.name for field in self.form.refresh()]
        if self.form_fieldsets:
            self.form_columns = []
            for fieldset_item in self.form_fieldsets:
                self.form_columns = self.form_columns + list(fieldset_item[1].get('fields'))
        elif not self.form_columns:
            self.form_columns = list_cols

    def form_get(self, form):
        if False:
            while True:
                i = 10
        '\n        Override this method to implement your form processing\n        '

    def form_post(self, form):
        if False:
            return 10
        '\n        Override this method to implement your form processing\n\n        :param form: WTForm form\n\n        Return None or a flask response to render\n        a custom template or redirect the user\n        '

    def _get_edit_widget(self, form=None, exclude_cols=None, widgets=None):
        if False:
            print('Hello World!')
        exclude_cols = exclude_cols or []
        widgets = widgets or {}
        widgets['edit'] = self.edit_widget(route_base=self.route_base, form=form, include_cols=self.form_columns, exclude_cols=exclude_cols, fieldsets=self.form_fieldsets)
        return widgets

class BaseModelView(BaseView):
    """
    The base class of ModelView and ChartView, all properties are inherited
    Customize ModelView and ChartView overriding this properties

    This class supports all the basics for query
    """
    datamodel = None
    '\n        Your sqla model you must initialize it like::\n\n            class MyView(ModelView):\n                datamodel = SQLAInterface(MyTable)\n    '
    title = 'Title'
    search_columns = None
    "\n        List with allowed search columns, if not provided\n        all possible search columns will be used\n        If you want to limit the search (*filter*) columns possibilities,\n        define it with a list of column names from your model::\n\n            class MyView(ModelView):\n                datamodel = SQLAInterface(MyTable)\n                search_columns = ['name','address']\n\n    "
    search_exclude_columns = None
    '\n        List with columns to exclude from search.\n        Search includes all possible columns by default\n    '
    search_form_extra_fields = None
    "\n        A dictionary containing column names and a WTForm\n        Form fields to be added to the search form, these fields do not\n        exist on the model itself ex::\n\n        search_form_extra_fields = {'some_col':BooleanField('Some Col', default=False)}\n\n    "
    search_form_query_rel_fields = None
    "\n        Add Customized query for related fields on search form.\n        Assign a dictionary where the keys are the column names of\n        the related models to filter, the value for each key, is a list of lists with the\n        same format as base_filter\n        {'relation col name':[['Related model col',FilterClass,'Filter Value'],...],...}\n        Add a custom filter to form related fields::\n\n            class ContactModelView(ModelView):\n                datamodel = SQLAModel(Contact, db.session)\n                search_form_query_rel_fields = {'group':[['name',FilterStartsWith,'W']]}\n\n    "
    label_columns = None
    "\n        Dictionary of labels for your columns,\n        override this if you want different pretify labels\n\n        example (will just override the label for name column)::\n\n            class MyView(ModelView):\n                datamodel = SQLAInterface(MyTable)\n                label_columns = {'name':'My Name Label Override'}\n\n    "
    search_form = None
    ' To implement your own add WTF form for Search '
    base_filters = None
    "\n        Filter the view use: [['column_name',BaseFilter,'value'],]\n\n        example::\n\n            def get_user():\n                return g.user\n\n            class MyView(ModelView):\n                datamodel = SQLAInterface(MyTable)\n                base_filters = [['created_by', FilterEqualFunction, get_user],\n                                ['name', FilterStartsWith, 'a']]\n\n    "
    base_order = None
    "\n        Use this property to set default ordering for lists ('col_name','asc|desc')::\n\n            class MyView(ModelView):\n                datamodel = SQLAInterface(MyTable)\n                base_order = ('my_column_name','asc')\n\n    "
    search_widget = SearchWidget
    ' Search widget you can override with your own '
    _base_filters = None
    ' Internal base Filter from class Filters will always filter view '
    _filters = None
    ' Filters object will calculate all possible filter types\n    based on search_columns '

    def __init__(self, **kwargs):
        if False:
            return 10
        '\n        Constructor\n        '
        datamodel = kwargs.get('datamodel', None)
        if datamodel:
            self.datamodel = datamodel
        self._init_properties()
        self._init_forms()
        self._init_titles()
        super(BaseModelView, self).__init__(**kwargs)

    def _gen_labels_columns(self, list_columns):
        if False:
            print('Hello World!')
        '\n        Auto generates pretty label_columns from list of columns\n        '
        for col in list_columns:
            if not self.label_columns.get(col):
                self.label_columns[col] = self._prettify_column(col)

    def _init_titles(self):
        if False:
            i = 10
            return i + 15
        pass

    def _init_properties(self):
        if False:
            return 10
        self.label_columns = self.label_columns or {}
        self.base_filters = self.base_filters or []
        self.search_exclude_columns = self.search_exclude_columns or []
        self.search_columns = self.search_columns or []
        self._base_filters = self.datamodel.get_filters().add_filter_list(self.base_filters)
        list_cols = self.datamodel.get_columns_list()
        search_columns = self.datamodel.get_search_columns_list()
        if not self.search_columns:
            self.search_columns = [x for x in search_columns if x not in self.search_exclude_columns]
        self._gen_labels_columns(list_cols)
        self._filters = self.datamodel.get_filters(self.search_columns)

    def _init_forms(self):
        if False:
            while True:
                i = 10
        conv = GeneralModelConverter(self.datamodel)
        if not self.search_form:
            self.search_form = conv.create_form(self.label_columns, self.search_columns, extra_fields=self.search_form_extra_fields, filter_rel_fields=self.search_form_query_rel_fields)

    def _get_search_widget(self, form=None, exclude_cols=None, widgets=None):
        if False:
            return 10
        exclude_cols = exclude_cols or []
        widgets = widgets or {}
        widgets['search'] = self.search_widget(route_base=self.route_base, form=form, include_cols=self.search_columns, exclude_cols=exclude_cols, filters=self._filters)
        return widgets

    def _label_columns_json(self):
        if False:
            return 10
        '\n        Prepares dict with labels to be JSON serializable\n        '
        ret = {}
        for (key, value) in list(self.label_columns.items()):
            ret[key] = as_unicode(value.encode('UTF-8'))
        return ret

class BaseCRUDView(BaseModelView):
    """
    The base class for ModelView, all properties are inherited
    Customize ModelView overriding this properties
    """
    related_views = None
    '\n        List with ModelView classes\n        Will be displayed related with this one using relationship sqlalchemy property::\n\n            class MyView(ModelView):\n                datamodel = SQLAModel(Group, db.session)\n                related_views = [MyOtherRelatedView]\n\n    '
    _related_views = None
    ' internal list with ref to instantiated view classes '
    list_title = ''
    " List Title, if not configured the default is 'List ' with pretty model name "
    show_title = ''
    " Show Title , if not configured the default is 'Show ' with pretty model name "
    add_title = ''
    " Add Title , if not configured the default is 'Add ' with pretty model name "
    edit_title = ''
    " Edit Title , if not configured the default is 'Edit ' with pretty model name "
    list_columns = None
    "\n        A list of columns (or model's methods) to be displayed on the list view.\n        Use it to control the order of the display\n    "
    show_columns = None
    "\n        A list of columns (or model's methods) to be displayed on the show view.\n        Use it to control the order of the display\n    "
    add_columns = None
    "\n        A list of columns (or model's methods) to be displayed on the add form view.\n        Use it to control the order of the display\n    "
    edit_columns = None
    "\n        A list of columns (or model's methods) to be displayed on the edit form view.\n        Use it to control the order of the display\n    "
    show_exclude_columns = None
    '\n       A list of columns to exclude from the show view.\n       By default all columns are included.\n    '
    add_exclude_columns = None
    '\n       A list of columns to exclude from the add form.\n       By default all columns are included.\n    '
    edit_exclude_columns = None
    '\n       A list of columns to exclude from the edit form.\n        By default all columns are included.\n    '
    order_columns = None
    ' Allowed order columns '
    page_size = 25
    '\n        Use this property to change default page size\n    '
    show_fieldsets = None
    "\n        show fieldsets django style [(<'TITLE'|None>, {'fields':[<F1>,<F2>,...]}),....]\n\n        ::\n\n            class MyView(ModelView):\n                datamodel = SQLAModel(MyTable, db.session)\n\n                show_fieldsets = [\n                    ('Summary', {\n                        'fields': [\n                            'name',\n                            'address',\n                            'group'\n                            ]\n                        }\n                    ),\n                    ('Personal Info', {\n                        'fields': [\n                            'birthday',\n                            'personal_phone'\n                            ],\n                        'expanded':False\n                        }\n                    ),\n                ]\n\n    "
    add_fieldsets = None
    '\n        add fieldsets django style (look at show_fieldsets for an example)\n    '
    edit_fieldsets = None
    '\n        edit fieldsets django style (look at show_fieldsets for an example)\n    '
    description_columns = None
    "\n        Dictionary with column descriptions that will be shown on the forms::\n\n            class MyView(ModelView):\n                datamodel = SQLAModel(MyTable, db.session)\n\n                description_columns = {\n                    'name': 'your models name column',\n                    'address': 'the address column'\n                }\n    "
    validators_columns = None
    ' Dictionary to add your own validators for forms '
    formatters_columns = None
    " Dictionary of formatter used to format the display of columns\n\n        formatters_columns = {'some_date_col': lambda x: x.isoformat() }\n    "
    add_form_extra_fields = None
    "\n        A dictionary containing column names and a WTForm\n        Form fields to be added to the Add form, these fields do not\n        exist on the model itself ex::\n\n        add_form_extra_fields = {'some_col':BooleanField('Some Col', default=False)}\n\n    "
    edit_form_extra_fields = None
    ' Dictionary to add extra fields to the Edit form using this property '
    add_form_query_rel_fields = None
    "\n        Add Customized query for related fields to add form.\n        Assign a dictionary where the keys are the column names of\n        the related models to filter, the value for each key, is a list of lists with the\n        same format as base_filter\n        {\n            'relation col name':\n                [['Related model col', FilterClass, 'Filter Value'],...],...\n        }\n        Add a custom filter to form related fields::\n\n            class ContactModelView(ModelView):\n                datamodel = SQLAModel(Contact, db.session)\n                add_form_query_rel_fields = {'group': [['name', FilterStartsWith, 'W']]}\n\n    "
    edit_form_query_rel_fields = None
    "\n        Add Customized query for related fields to edit form.\n        Assign a dictionary where the keys are the column names of\n        the related models to filter, the value for each key, is a list of lists with the\n        same format as base_filter\n        {\n            'relation col name':\n                [['Related model col', FilterClass, 'Filter Value'],...],...\n        }\n        Add a custom filter to form related fields::\n\n            class ContactModelView(ModelView):\n                datamodel = SQLAModel(Contact, db.session)\n                edit_form_query_rel_fields = {'group':[['name',FilterStartsWith,'W']]}\n\n    "
    add_form = None
    ' To implement your own, assign WTF form for Add '
    edit_form = None
    ' To implement your own, assign WTF form for Edit '
    list_template = 'appbuilder/general/model/list.html'
    ' Your own add jinja2 template for list '
    edit_template = 'appbuilder/general/model/edit.html'
    ' Your own add jinja2 template for edit '
    add_template = 'appbuilder/general/model/add.html'
    ' Your own add jinja2 template for add '
    show_template = 'appbuilder/general/model/show.html'
    ' Your own add jinja2 template for show '
    list_widget = ListWidget
    ' List widget override '
    edit_widget = FormWidget
    ' Edit widget override '
    add_widget = FormWidget
    ' Add widget override '
    show_widget = ShowWidget
    ' Show widget override '
    actions = None

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(BaseCRUDView, self).__init__(**kwargs)
        self.actions = {}
        for attr_name in dir(self):
            func = getattr(self, attr_name)
            if hasattr(func, '_action'):
                action = ActionItem(*func._action, func=func)
                permission_name = action.name
                if self.method_permission_name.get(attr_name):
                    if not self.previous_method_permission_name.get(attr_name):
                        self.previous_method_permission_name[attr_name] = action.name
                    permission_name = PERMISSION_PREFIX + self.method_permission_name.get(attr_name)
                if permission_name not in self.base_permissions:
                    self.base_permissions.append(permission_name)
                self.actions[action.name] = action

    def _init_forms(self):
        if False:
            i = 10
            return i + 15
        '\n        Init forms for Add and Edit\n        '
        super(BaseCRUDView, self)._init_forms()
        conv = GeneralModelConverter(self.datamodel)
        if not self.add_form:
            self.add_form = conv.create_form(self.label_columns, self.add_columns, self.description_columns, self.validators_columns, self.add_form_extra_fields, self.add_form_query_rel_fields)
        if not self.edit_form:
            self.edit_form = conv.create_form(self.label_columns, self.edit_columns, self.description_columns, self.validators_columns, self.edit_form_extra_fields, self.edit_form_query_rel_fields)

    def _init_titles(self):
        if False:
            return 10
        '\n        Init Titles if not defined\n        '
        super(BaseCRUDView, self)._init_titles()
        class_name = self.datamodel.model_name
        if not self.list_title:
            self.list_title = 'List ' + self._prettify_name(class_name)
        if not self.add_title:
            self.add_title = 'Add ' + self._prettify_name(class_name)
        if not self.edit_title:
            self.edit_title = 'Edit ' + self._prettify_name(class_name)
        if not self.show_title:
            self.show_title = 'Show ' + self._prettify_name(class_name)
        self.title = self.list_title

    def _init_properties(self):
        if False:
            return 10
        '\n        Init Properties\n        '
        super(BaseCRUDView, self)._init_properties()
        self.related_views = self.related_views or []
        self._related_views = self._related_views or []
        self.description_columns = self.description_columns or {}
        self.validators_columns = self.validators_columns or {}
        self.formatters_columns = self.formatters_columns or {}
        self.add_form_extra_fields = self.add_form_extra_fields or {}
        self.edit_form_extra_fields = self.edit_form_extra_fields or {}
        self.show_exclude_columns = self.show_exclude_columns or []
        self.add_exclude_columns = self.add_exclude_columns or []
        self.edit_exclude_columns = self.edit_exclude_columns or []
        list_cols = self.datamodel.get_user_columns_list()
        self.list_columns = self.list_columns or [list_cols[0]]
        self._gen_labels_columns(self.list_columns)
        self.order_columns = self.order_columns or self.datamodel.get_order_columns_list(list_columns=self.list_columns)
        if self.show_fieldsets:
            self.show_columns = []
            for fieldset_item in self.show_fieldsets:
                self.show_columns = self.show_columns + list(fieldset_item[1].get('fields'))
        elif not self.show_columns:
            self.show_columns = [x for x in list_cols if x not in self.show_exclude_columns]
        if self.add_fieldsets:
            self.add_columns = []
            for fieldset_item in self.add_fieldsets:
                self.add_columns = self.add_columns + list(fieldset_item[1].get('fields'))
        elif not self.add_columns:
            self.add_columns = [x for x in list_cols if x not in self.add_exclude_columns]
        if self.edit_fieldsets:
            self.edit_columns = []
            for fieldset_item in self.edit_fieldsets:
                self.edit_columns = self.edit_columns + list(fieldset_item[1].get('fields'))
        elif not self.edit_columns:
            self.edit_columns = [x for x in list_cols if x not in self.edit_exclude_columns]
    '\n    -----------------------------------------------------\n            GET WIDGETS SECTION\n    -----------------------------------------------------\n    '

    def _get_related_view_widget(self, item, related_view, order_column='', order_direction='', page=None, page_size=None):
        if False:
            return 10
        fk = related_view.datamodel.get_related_fk(self.datamodel.obj)
        filters = related_view.datamodel.get_filters()
        if related_view.datamodel.is_relation_many_to_one(fk):
            filters.add_filter_related_view(fk, self.datamodel.FilterRelationOneToManyEqual, self.datamodel.get_pk_value(item))
        elif related_view.datamodel.is_relation_many_to_many(fk):
            filters.add_filter_related_view(fk, self.datamodel.FilterRelationManyToManyEqual, self.datamodel.get_pk_value(item))
        else:
            if isclass(related_view) and issubclass(related_view, BaseView):
                name = related_view.__name__
            else:
                name = related_view.__class__.__name__
            log.error("Can't find relation on related view %s", name)
            return None
        return related_view._get_view_widget(filters=filters, order_column=order_column, order_direction=order_direction, page=page, page_size=page_size)

    def _get_related_views_widgets(self, item, orders=None, pages=None, page_sizes=None, widgets=None, **args):
        if False:
            i = 10
            return i + 15
        "\n        :return:\n            Returns a dict with 'related_views' key with a list of\n            Model View widgets\n        "
        widgets = widgets or {}
        widgets['related_views'] = []
        for view in self._related_views:
            if orders.get(view.__class__.__name__):
                (order_column, order_direction) = orders.get(view.__class__.__name__)
            else:
                (order_column, order_direction) = ('', '')
            widgets['related_views'].append(self._get_related_view_widget(item, view, order_column, order_direction, page=pages.get(view.__class__.__name__), page_size=page_sizes.get(view.__class__.__name__)))
        return widgets

    def _get_view_widget(self, **kwargs):
        if False:
            while True:
                i = 10
        '\n        :return:\n            Returns a Model View widget\n        '
        return self._get_list_widget(**kwargs).get('list')

    def _get_list_widget(self, filters, actions=None, order_column='', order_direction='', page=None, page_size=None, widgets=None, **args):
        if False:
            return 10
        'get joined base filter and current active filter for query'
        widgets = widgets or {}
        actions = actions or self.actions
        page_size = page_size or self.page_size
        if not order_column and self.base_order:
            (order_column, order_direction) = self.base_order
        joined_filters = filters.get_joined_filters(self._base_filters)
        (count, lst) = self.datamodel.query(joined_filters, order_column, order_direction, page=page, page_size=page_size)
        pks = self.datamodel.get_keys(lst)
        pks = [self._serialize_pk_if_composite(pk) for pk in pks]
        widgets['list'] = self.list_widget(label_columns=self.label_columns, include_columns=self.list_columns, value_columns=self.datamodel.get_values(lst, self.list_columns), order_columns=self.order_columns, formatters_columns=self.formatters_columns, page=page, page_size=page_size, count=count, pks=pks, actions=actions, filters=filters, modelview_name=self.__class__.__name__)
        return widgets

    def _get_show_widget(self, pk, item, widgets=None, actions=None, show_fieldsets=None):
        if False:
            print('Hello World!')
        widgets = widgets or {}
        actions = actions or self.actions
        show_fieldsets = show_fieldsets or self.show_fieldsets
        widgets['show'] = self.show_widget(pk=pk, label_columns=self.label_columns, include_columns=self.show_columns, value_columns=self.datamodel.get_values_item(item, self.show_columns), formatters_columns=self.formatters_columns, actions=actions, fieldsets=show_fieldsets, modelview_name=self.__class__.__name__)
        return widgets

    def _get_add_widget(self, form, exclude_cols=None, widgets=None):
        if False:
            i = 10
            return i + 15
        exclude_cols = exclude_cols or []
        widgets = widgets or {}
        widgets['add'] = self.add_widget(form=form, include_cols=self.add_columns, exclude_cols=exclude_cols, fieldsets=self.add_fieldsets)
        return widgets

    def _get_edit_widget(self, form, exclude_cols=None, widgets=None):
        if False:
            return 10
        exclude_cols = exclude_cols or []
        widgets = widgets or {}
        widgets['edit'] = self.edit_widget(form=form, include_cols=self.edit_columns, exclude_cols=exclude_cols, fieldsets=self.edit_fieldsets)
        return widgets

    def get_uninit_inner_views(self):
        if False:
            print('Hello World!')
        '\n        Will return a list with views that need to be initialized.\n        Normally related_views from ModelView\n        '
        return self.related_views

    def get_init_inner_views(self):
        if False:
            print('Hello World!')
        '\n        Get the list of related ModelViews after they have been initialized\n        '
        return self._related_views
    '\n    -----------------------------------------------------\n            CRUD functions behaviour\n    -----------------------------------------------------\n    '

    def _list(self):
        if False:
            while True:
                i = 10
        '\n        list function logic, override to implement different logic\n        returns list and search widget\n        '
        if get_order_args().get(self.__class__.__name__):
            (order_column, order_direction) = get_order_args().get(self.__class__.__name__)
        else:
            (order_column, order_direction) = ('', '')
        page = get_page_args().get(self.__class__.__name__)
        page_size = get_page_size_args().get(self.__class__.__name__)
        get_filter_args(self._filters)
        widgets = self._get_list_widget(filters=self._filters, order_column=order_column, order_direction=order_direction, page=page, page_size=page_size)
        form = self.search_form.refresh()
        self.update_redirect()
        return self._get_search_widget(form=form, widgets=widgets)

    def _show(self, pk):
        if False:
            return 10
        '\n        show function logic, override to implement different logic\n        returns show and related list widget\n        '
        pages = get_page_args()
        page_sizes = get_page_size_args()
        orders = get_order_args()
        item = self.datamodel.get(pk, self._base_filters)
        if not item:
            abort(404)
        widgets = self._get_show_widget(pk, item)
        self.update_redirect()
        return self._get_related_views_widgets(item, orders=orders, pages=pages, page_sizes=page_sizes, widgets=widgets)

    def _add(self):
        if False:
            while True:
                i = 10
        '\n        Add function logic, override to implement different logic\n        returns add widget or None\n        '
        is_valid_form = True
        get_filter_args(self._filters, disallow_if_not_in_search=False)
        exclude_cols = self._filters.get_relation_cols()
        form = self.add_form.refresh()
        if request.method == 'POST':
            self._fill_form_exclude_cols(exclude_cols, form)
            if form.validate():
                self.process_form(form, True)
                item = self.datamodel.obj()
                try:
                    form.populate_obj(item)
                    self.pre_add(item)
                except Exception as e:
                    flash(str(e), 'danger')
                else:
                    if self.datamodel.add(item):
                        self.post_add(item)
                    flash(*self.datamodel.message)
                finally:
                    return None
            else:
                is_valid_form = False
        if is_valid_form:
            self.update_redirect()
        return self._get_add_widget(form=form, exclude_cols=exclude_cols)

    def _edit(self, pk):
        if False:
            for i in range(10):
                print('nop')
        '\n        Edit function logic, override to implement different logic\n        returns Edit widget and related list or None\n        '
        is_valid_form = True
        pages = get_page_args()
        page_sizes = get_page_size_args()
        orders = get_order_args()
        get_filter_args(self._filters, disallow_if_not_in_search=False)
        exclude_cols = self._filters.get_relation_cols()
        item = self.datamodel.get(pk, self._base_filters)
        if not item:
            abort(404)
        pk = self.datamodel.get_pk_value(item)
        if request.method == 'POST':
            form = self.edit_form.refresh(request.form)
            self._fill_form_exclude_cols(exclude_cols, form)
            form._id = pk
            if form.validate():
                self.process_form(form, False)
                try:
                    form.populate_obj(item)
                    self.pre_update(item)
                except Exception as e:
                    flash(str(e), 'danger')
                else:
                    if self.datamodel.edit(item):
                        self.post_update(item)
                    flash(*self.datamodel.message)
                finally:
                    return None
            else:
                is_valid_form = False
        else:
            form = self.edit_form.refresh(obj=item)
            self.prefill_form(form, pk)
        widgets = self._get_edit_widget(form=form, exclude_cols=exclude_cols)
        widgets = self._get_related_views_widgets(item, filters={}, orders=orders, pages=pages, page_sizes=page_sizes, widgets=widgets)
        if is_valid_form:
            self.update_redirect()
        return widgets

    def _delete(self, pk):
        if False:
            print('Hello World!')
        '\n        Delete function logic, override to implement different logic\n        deletes the record with primary_key = pk\n\n        :param pk:\n            record primary key to delete\n        '
        item = self.datamodel.get(pk, self._base_filters)
        if not item:
            abort(404)
        try:
            self.pre_delete(item)
        except Exception as e:
            flash(str(e), 'danger')
        else:
            if self.datamodel.delete(item):
                self.post_delete(item)
            flash(*self.datamodel.message)
            self.update_redirect()
    '\n    ------------------------------------------------\n                HELPER FUNCTIONS\n    ------------------------------------------------\n    '

    def _serialize_pk_if_composite(self, pk):
        if False:
            return 10

        def date_serializer(obj):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(obj, datetime):
                return {'_type': 'datetime', 'value': obj.isoformat()}
            elif isinstance(obj, date):
                return {'_type': 'date', 'value': obj.isoformat()}
        if self.datamodel.is_pk_composite():
            try:
                pk = json.dumps(pk, default=date_serializer)
            except Exception:
                pass
        return pk

    def _deserialize_pk_if_composite(self, pk):
        if False:
            return 10

        def date_deserializer(obj):
            if False:
                for i in range(10):
                    print('nop')
            if '_type' not in obj:
                return obj
            from dateutil import parser
            if obj['_type'] == 'datetime':
                return parser.parse(obj['value'])
            elif obj['_type'] == 'date':
                return parser.parse(obj['value']).date()
            return obj
        if self.datamodel.is_pk_composite():
            try:
                pk = json.loads(pk, object_hook=date_deserializer)
            except Exception:
                pass
        return pk

    def _fill_form_exclude_cols(self, exclude_cols, form):
        if False:
            print('Hello World!')
        '\n        fill the form with the suppressed cols, generated from exclude_cols\n        '
        for filter_key in exclude_cols:
            filter_value = self._filters.get_filter_value(filter_key)
            rel_obj = self.datamodel.get_related_obj(filter_key, filter_value)
            if hasattr(form, filter_key):
                field = getattr(form, filter_key)
                field.data = rel_obj

    def is_get_mutation_allowed(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Check is mutations on HTTP GET methods are allowed.\n        Always called on a request\n        '
        if current_app.config.get('FAB_ALLOW_GET_UNSAFE_MUTATIONS', False):
            return True
        return not (request.method == 'GET' and self.appbuilder.app.extensions.get('csrf'))

    def prefill_form(self, form, pk):
        if False:
            while True:
                i = 10
        '\n        Override this, will be called only if the current action is rendering\n        an edit form (a GET request), and is used to perform additional action to\n        prefill the form.\n\n        This is useful when you have added custom fields that depend on the\n        database contents. Fields that were added by name of a normal column\n        or relationship should work out of the box.\n\n        example::\n\n            def prefill_form(self, form, pk):\n                if form.email.data:\n                    form.email_confirmation.data = form.email.data\n        '

    def process_form(self, form, is_created):
        if False:
            print('Hello World!')
        "\n        Override this, will be called only if the current action is submitting\n        a create/edit form (a POST request), and is used to perform additional\n        action before the form is used to populate the item.\n\n        By default does nothing.\n\n        example::\n\n            def process_form(self, form, is_created):\n                if not form.owner:\n                    form.owner.data = 'n/a'\n        "

    def pre_update(self, item):
        if False:
            return 10
        '\n        Override this, this method is called before the update takes place.\n        If an exception is raised by this method,\n        the message is shown to the user and the update operation is\n        aborted. Because of this behavior, it can be used as a way to\n        implement more complex logic around updates. For instance\n        allowing only the original creator of the object to update it.\n        '

    def post_update(self, item):
        if False:
            while True:
                i = 10
        '\n        Override this, will be called after update\n        '

    def pre_add(self, item):
        if False:
            return 10
        '\n        Override this, will be called before add.\n        If an exception is raised by this method,\n        the message is shown to the user and the add operation is aborted.\n        '

    def post_add(self, item):
        if False:
            print('Hello World!')
        '\n        Override this, will be called after update\n        '

    def pre_delete(self, item):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override this, will be called before delete\n        If an exception is raised by this method,\n        the message is shown to the user and the delete operation is\n        aborted. Because of this behavior, it can be used as a way to\n        implement more complex logic around deletes. For instance\n        allowing only the original creator of the object to delete it.\n        '

    def post_delete(self, item):
        if False:
            while True:
                i = 10
        '\n        Override this, will be called after delete\n        '