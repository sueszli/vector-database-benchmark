from __future__ import annotations
import logging
from functools import reduce
from typing import TYPE_CHECKING
from flask import Blueprint, current_app, url_for
from flask_appbuilder import __version__
from flask_appbuilder.babel.manager import BabelManager
from flask_appbuilder.const import LOGMSG_ERR_FAB_ADD_PERMISSION_MENU, LOGMSG_ERR_FAB_ADD_PERMISSION_VIEW, LOGMSG_ERR_FAB_ADDON_IMPORT, LOGMSG_ERR_FAB_ADDON_PROCESS, LOGMSG_INF_FAB_ADD_VIEW, LOGMSG_INF_FAB_ADDON_ADDED, LOGMSG_WAR_FAB_VIEW_EXISTS
from flask_appbuilder.filters import TemplateFilters
from flask_appbuilder.menu import Menu
from flask_appbuilder.views import IndexView, UtilView
from airflow import settings
from airflow.configuration import conf
from airflow.www.extensions.init_auth_manager import get_auth_manager, init_auth_manager
if TYPE_CHECKING:
    from flask import Flask
    from flask_appbuilder import BaseView
    from flask_appbuilder.security.manager import BaseSecurityManager
    from sqlalchemy.orm import Session
log = logging.getLogger('flask_appbuilder.base')

def dynamic_class_import(class_path):
    if False:
        while True:
            i = 10
    '\n    Will dynamically import a class from a string path.\n\n    :param class_path: string with class path\n    :return: class\n    '
    try:
        tmp = class_path.split('.')
        module_path = '.'.join(tmp[0:-1])
        package = __import__(module_path)
        return reduce(getattr, tmp[1:], package)
    except Exception as e:
        log.exception(e)
        log.error(LOGMSG_ERR_FAB_ADDON_IMPORT, class_path, e)

class AirflowAppBuilder:
    """
    This is the base class for all the framework.

    This is where you will register all your views
    and create the menu structure.
    Will hold your flask app object, all your views, and security classes.
    Initialize your application like this for SQLAlchemy::
        from flask import Flask
        from flask_appbuilder import SQLA, AppBuilder
        app = Flask(__name__)
        app.config.from_object('config')
        db = SQLA(app)
        appbuilder = AppBuilder(app, db.session)
    When using MongoEngine::
        from flask import Flask
        from flask_appbuilder import AppBuilder
        from flask_appbuilder.security.mongoengine.manager import SecurityManager
        from flask_mongoengine import MongoEngine
        app = Flask(__name__)
        app.config.from_object('config')
        dbmongo = MongoEngine(app)
        appbuilder = AppBuilder(app)
    You can also create everything as an application factory.
    """
    baseviews: list[BaseView | Session] = []
    app = None
    session = None
    sm: BaseSecurityManager
    bm = None
    addon_managers: dict
    _addon_managers: list
    menu = None
    indexview = None
    static_folder = None
    static_url_path = None
    template_filters = None

    def __init__(self, app=None, session: Session | None=None, menu=None, indexview=None, base_template='airflow/main.html', static_folder='static/appbuilder', static_url_path='/appbuilder', update_perms=conf.getboolean('webserver', 'UPDATE_FAB_PERMS'), auth_rate_limited=conf.getboolean('webserver', 'AUTH_RATE_LIMITED', fallback=True), auth_rate_limit=conf.get('webserver', 'AUTH_RATE_LIMIT', fallback='5 per 40 second')):
        if False:
            for i in range(10):
                print('nop')
        '\n        App-builder constructor.\n\n        :param app:\n            The flask app object\n        :param session:\n            The SQLAlchemy session object\n        :param menu:\n            optional, a previous constructed menu\n        :param indexview:\n            optional, your customized indexview\n        :param static_folder:\n            optional, your override for the global static folder\n        :param static_url_path:\n            optional, your override for the global static url path\n        :param update_perms:\n            optional, update permissions flag (Boolean) you can use\n            FAB_UPDATE_PERMS config key also\n        :param auth_rate_limited:\n            optional, rate limit authentication attempts if set to True (defaults to True)\n        :param auth_rate_limit:\n            optional, rate limit authentication attempts configuration (defaults "to 5 per 40 second")\n        '
        self.baseviews = []
        self._addon_managers = []
        self.addon_managers = {}
        self.menu = menu
        self.base_template = base_template
        self.indexview = indexview
        self.static_folder = static_folder
        self.static_url_path = static_url_path
        self.app = app
        self.update_perms = update_perms
        self.auth_rate_limited = auth_rate_limited
        self.auth_rate_limit = auth_rate_limit
        if app is not None:
            self.init_app(app, session)

    def init_app(self, app, session):
        if False:
            print('Hello World!')
        '\n        Will initialize the Flask app, supporting the app factory pattern.\n\n        :param app:\n        :param session: The SQLAlchemy session\n        '
        app.config.setdefault('APP_NAME', 'F.A.B.')
        app.config.setdefault('APP_THEME', '')
        app.config.setdefault('APP_ICON', '')
        app.config.setdefault('LANGUAGES', {'en': {'flag': 'gb', 'name': 'English'}})
        app.config.setdefault('ADDON_MANAGERS', [])
        app.config.setdefault('RATELIMIT_ENABLED', self.auth_rate_limited)
        app.config.setdefault('FAB_API_MAX_PAGE_SIZE', 100)
        app.config.setdefault('FAB_BASE_TEMPLATE', self.base_template)
        app.config.setdefault('FAB_STATIC_FOLDER', self.static_folder)
        app.config.setdefault('FAB_STATIC_URL_PATH', self.static_url_path)
        app.config.setdefault('AUTH_RATE_LIMITED', self.auth_rate_limited)
        app.config.setdefault('AUTH_RATE_LIMIT', self.auth_rate_limit)
        self.app = app
        self.base_template = app.config.get('FAB_BASE_TEMPLATE', self.base_template)
        self.static_folder = app.config.get('FAB_STATIC_FOLDER', self.static_folder)
        self.static_url_path = app.config.get('FAB_STATIC_URL_PATH', self.static_url_path)
        _index_view = app.config.get('FAB_INDEX_VIEW', None)
        if _index_view is not None:
            self.indexview = dynamic_class_import(_index_view)
        else:
            self.indexview = self.indexview or IndexView
        _menu = app.config.get('FAB_MENU', None)
        if _menu is not None:
            self.menu = dynamic_class_import(_menu)
        else:
            self.menu = self.menu or Menu()
        if self.update_perms:
            self.update_perms = app.config.get('FAB_UPDATE_PERMS', True)
        self._addon_managers = app.config['ADDON_MANAGERS']
        self.session = session
        auth_manager = init_auth_manager(app, self)
        self.sm = auth_manager.security_manager
        self.bm = BabelManager(self)
        self._add_global_static()
        self._add_global_filters()
        app.before_request(self.sm.before_request)
        self._add_admin_views()
        self._add_addon_views()
        if self.app:
            self._add_menu_permissions()
        else:
            self.post_init()
        self._init_extension(app)
        self._swap_url_filter()

    def _init_extension(self, app):
        if False:
            for i in range(10):
                print('nop')
        app.appbuilder = self
        if not hasattr(app, 'extensions'):
            app.extensions = {}
        app.extensions['appbuilder'] = self

    def _swap_url_filter(self):
        if False:
            for i in range(10):
                print('nop')
        'Use our url filtering util function so there is consistency between FAB and Airflow routes.'
        from flask_appbuilder.security import views as fab_sec_views
        from airflow.www.views import get_safe_url
        fab_sec_views.get_safe_redirect = get_safe_url

    def post_init(self):
        if False:
            for i in range(10):
                print('nop')
        for baseview in self.baseviews:
            self._check_and_init(baseview)
            if baseview.__class__.__name__ not in self.get_app.blueprints.keys():
                self.register_blueprint(baseview)
        self.add_permissions()

    @property
    def get_app(self):
        if False:
            while True:
                i = 10
        '\n        Get current or configured flask app.\n\n        :return: Flask App\n        '
        if self.app:
            return self.app
        else:
            return current_app

    @property
    def get_session(self):
        if False:
            return 10
        '\n        Get the current sqlalchemy session.\n\n        :return: SQLAlchemy Session\n        '
        return self.session

    @property
    def app_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the App name.\n\n        :return: String with app name\n        '
        return self.get_app.config['APP_NAME']

    @property
    def app_theme(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the App theme name.\n\n        :return: String app theme name\n        '
        return self.get_app.config['APP_THEME']

    @property
    def app_icon(self):
        if False:
            print('Hello World!')
        '\n        Get the App icon location.\n\n        :return: String with relative app icon location\n        '
        return self.get_app.config['APP_ICON']

    @property
    def languages(self):
        if False:
            print('Hello World!')
        return self.get_app.config['LANGUAGES']

    @property
    def version(self):
        if False:
            while True:
                i = 10
        '\n        Get the current F.A.B. version.\n\n        :return: String with the current F.A.B. version\n        '
        return __version__

    def _add_global_filters(self):
        if False:
            return 10
        self.template_filters = TemplateFilters(self.get_app, self.sm)

    def _add_global_static(self):
        if False:
            print('Hello World!')
        bp = Blueprint('appbuilder', 'flask_appbuilder.base', url_prefix='/static', template_folder='templates', static_folder=self.static_folder, static_url_path=self.static_url_path)
        self.get_app.register_blueprint(bp)

    def _add_admin_views(self):
        if False:
            while True:
                i = 10
        'Register indexview, utilview (back function), babel views and Security views.'
        self.indexview = self._check_and_init(self.indexview)
        self.add_view_no_menu(self.indexview)
        self.add_view_no_menu(UtilView())
        self.bm.register_views()
        self.sm.register_views()

    def _add_addon_views(self):
        if False:
            i = 10
            return i + 15
        'Register declared addons.'
        for addon in self._addon_managers:
            addon_class = dynamic_class_import(addon)
            if addon_class:
                addon_class = addon_class(self)
                try:
                    addon_class.pre_process()
                    addon_class.register_views()
                    addon_class.post_process()
                    self.addon_managers[addon] = addon_class
                    log.info(LOGMSG_INF_FAB_ADDON_ADDED, addon)
                except Exception as e:
                    log.exception(e)
                    log.error(LOGMSG_ERR_FAB_ADDON_PROCESS, addon, e)

    def _check_and_init(self, baseview):
        if False:
            while True:
                i = 10
        if hasattr(baseview, 'datamodel'):
            baseview.datamodel.session = self.session
        if hasattr(baseview, '__call__'):
            baseview = baseview()
        return baseview

    def add_view(self, baseview, name, href='', icon='', label='', category='', category_icon='', category_label='', menu_cond=None):
        if False:
            i = 10
            return i + 15
        'Add your views associated with menus using this method.\n\n        :param baseview:\n            A BaseView type class instantiated or not.\n            This method will instantiate the class for you if needed.\n        :param name:\n            The string name that identifies the menu.\n        :param href:\n            Override the generated href for the menu.\n            You can use an url string or an endpoint name\n            if non provided default_view from view will be set as href.\n        :param icon:\n            Font-Awesome icon name, optional.\n        :param label:\n            The label that will be displayed on the menu,\n            if absent param name will be used\n        :param category:\n            The menu category where the menu will be included,\n            if non provided the view will be accessible as a top menu.\n        :param category_icon:\n            Font-Awesome icon name for the category, optional.\n        :param category_label:\n            The label that will be displayed on the menu,\n            if absent param name will be used\n        :param menu_cond:\n            If a callable, :code:`menu_cond` will be invoked when\n            constructing the menu items. If it returns :code:`True`,\n            then this link will be a part of the menu. Otherwise, it\n            will not be included in the menu items. Defaults to\n            :code:`None`, meaning the item will always be present.\n\n        Examples::\n\n            appbuilder = AppBuilder(app, db)\n            # Register a view, rendering a top menu without icon.\n            appbuilder.add_view(MyModelView(), "My View")\n            # or not instantiated\n            appbuilder.add_view(MyModelView, "My View")\n            # Register a view, a submenu "Other View" from "Other" with a phone icon.\n            appbuilder.add_view(\n                MyOtherModelView,\n                "Other View",\n                icon=\'fa-phone\',\n                category="Others"\n            )\n            # Register a view, with category icon and translation.\n            appbuilder.add_view(\n                YetOtherModelView,\n                "Other View",\n                icon=\'fa-phone\',\n                label=_(\'Other View\'),\n                category="Others",\n                category_icon=\'fa-envelop\',\n                category_label=_(\'Other View\')\n            )\n            # Register a view whose menu item will be conditionally displayed\n            appbuilder.add_view(\n                YourFeatureView,\n                "Your Feature",\n                icon=\'fa-feature\',\n                label=_(\'Your Feature\'),\n                menu_cond=lambda: is_feature_enabled("your-feature"),\n            )\n            # Add a link\n            appbuilder.add_link("google", href="www.google.com", icon = "fa-google-plus")\n        '
        baseview = self._check_and_init(baseview)
        log.info(LOGMSG_INF_FAB_ADD_VIEW, baseview.__class__.__name__, name)
        if not self._view_exists(baseview):
            baseview.appbuilder = self
            self.baseviews.append(baseview)
            self._process_inner_views()
            if self.app:
                self.register_blueprint(baseview)
                self._add_permission(baseview)
                self.add_limits(baseview)
        self.add_link(name=name, href=href, icon=icon, label=label, category=category, category_icon=category_icon, category_label=category_label, baseview=baseview, cond=menu_cond)
        return baseview

    def add_link(self, name, href, icon='', label='', category='', category_icon='', category_label='', baseview=None, cond=None):
        if False:
            return 10
        'Add your own links to menu using this method.\n\n        :param name:\n            The string name that identifies the menu.\n        :param href:\n            Override the generated href for the menu.\n            You can use an url string or an endpoint name\n        :param icon:\n            Font-Awesome icon name, optional.\n        :param label:\n            The label that will be displayed on the menu,\n            if absent param name will be used\n        :param category:\n            The menu category where the menu will be included,\n            if non provided the view will be accessible as a top menu.\n        :param category_icon:\n            Font-Awesome icon name for the category, optional.\n        :param category_label:\n            The label that will be displayed on the menu,\n            if absent param name will be used\n        :param baseview:\n            A BaseView type class instantiated.\n        :param cond:\n            If a callable, :code:`cond` will be invoked when\n            constructing the menu items. If it returns :code:`True`,\n            then this link will be a part of the menu. Otherwise, it\n            will not be included in the menu items. Defaults to\n            :code:`None`, meaning the item will always be present.\n        '
        self.menu.add_link(name=name, href=href, icon=icon, label=label, category=category, category_icon=category_icon, category_label=category_label, baseview=baseview, cond=cond)
        if self.app:
            self._add_permissions_menu(name)
            if category:
                self._add_permissions_menu(category)

    def add_separator(self, category, cond=None):
        if False:
            i = 10
            return i + 15
        'Add a separator to the menu, you will sequentially create the menu.\n\n        :param category:\n            The menu category where the separator will be included.\n        :param cond:\n            If a callable, :code:`cond` will be invoked when\n            constructing the menu items. If it returns :code:`True`,\n            then this separator will be a part of the menu. Otherwise,\n            it will not be included in the menu items. Defaults to\n            :code:`None`, meaning the separator will always be present.\n        '
        self.menu.add_separator(category, cond=cond)

    def add_view_no_menu(self, baseview, endpoint=None, static_folder=None):
        if False:
            return 10
        '\n        Add your views without creating a menu.\n\n        :param baseview: A BaseView type class instantiated.\n        '
        baseview = self._check_and_init(baseview)
        log.info(LOGMSG_INF_FAB_ADD_VIEW, baseview.__class__.__name__, '')
        if not self._view_exists(baseview):
            baseview.appbuilder = self
            self.baseviews.append(baseview)
            self._process_inner_views()
            if self.app:
                self.register_blueprint(baseview, endpoint=endpoint, static_folder=static_folder)
                self._add_permission(baseview)
        else:
            log.warning(LOGMSG_WAR_FAB_VIEW_EXISTS, baseview.__class__.__name__)
        return baseview

    def security_cleanup(self):
        if False:
            return 10
        'Clean up security.\n\n        This method is useful if you have changed the name of your menus or\n        classes. Changing them leaves behind permissions that are not associated\n        with anything. You can use it always or just sometimes to perform a\n        security cleanup.\n\n        .. warning::\n\n            This deletes any permission that is no longer part of any registered\n            view or menu. Only invoke AFTER YOU HAVE REGISTERED ALL VIEWS.\n        '
        if not hasattr(self.sm, 'security_cleanup'):
            raise NotImplementedError('The auth manager used does not support security_cleanup method.')
        self.sm.security_cleanup(self.baseviews, self.menu)

    def security_converge(self, dry=False) -> dict:
        if False:
            for i in range(10):
                print('nop')
        'Migrates all permissions to the new names on all the Roles.\n\n        This method is useful when you use:\n\n        - ``class_permission_name``\n        - ``previous_class_permission_name``\n        - ``method_permission_name``\n        - ``previous_method_permission_name``\n\n        :param dry: If True will not change DB\n        :return: Dict with all computed necessary operations\n        '
        return self.sm.security_converge(self.baseviews, self.menu, dry)

    def get_url_for_login_with(self, next_url: str | None=None) -> str:
        if False:
            while True:
                i = 10
        return get_auth_manager().get_url_login(next_url=next_url)

    @property
    def get_url_for_login(self):
        if False:
            print('Hello World!')
        return get_auth_manager().get_url_login()

    @property
    def get_url_for_index(self):
        if False:
            return 10
        return url_for(f'{self.indexview.endpoint}.{self.indexview.default_view}')

    def get_url_for_locale(self, lang):
        if False:
            while True:
                i = 10
        return url_for(f'{self.bm.locale_view.endpoint}.{self.bm.locale_view.default_view}', locale=lang)

    def add_limits(self, baseview) -> None:
        if False:
            i = 10
            return i + 15
        if hasattr(baseview, 'limits'):
            self.sm.add_limit_view(baseview)

    def add_permissions(self, update_perms=False):
        if False:
            print('Hello World!')
        if self.update_perms or update_perms:
            for baseview in self.baseviews:
                self._add_permission(baseview, update_perms=update_perms)
            self._add_menu_permissions(update_perms=update_perms)

    def _add_permission(self, baseview, update_perms=False):
        if False:
            for i in range(10):
                print('nop')
        if self.update_perms or update_perms:
            try:
                self.sm.add_permissions_view(baseview.base_permissions, baseview.class_permission_name)
            except Exception as e:
                log.exception(e)
                log.error(LOGMSG_ERR_FAB_ADD_PERMISSION_VIEW, e)

    def _add_permissions_menu(self, name, update_perms=False):
        if False:
            while True:
                i = 10
        if self.update_perms or update_perms:
            try:
                self.sm.add_permissions_menu(name)
            except Exception as e:
                log.exception(e)
                log.error(LOGMSG_ERR_FAB_ADD_PERMISSION_MENU, e)

    def _add_menu_permissions(self, update_perms=False):
        if False:
            print('Hello World!')
        if self.update_perms or update_perms:
            for category in self.menu.get_list():
                self._add_permissions_menu(category.name, update_perms=update_perms)
                for item in category.childs:
                    if item.name != '-':
                        self._add_permissions_menu(item.name, update_perms=update_perms)

    def register_blueprint(self, baseview, endpoint=None, static_folder=None):
        if False:
            print('Hello World!')
        self.get_app.register_blueprint(baseview.create_blueprint(self, endpoint=endpoint, static_folder=static_folder))

    def _view_exists(self, view):
        if False:
            return 10
        return any((baseview.__class__ == view.__class__ for baseview in self.baseviews))

    def _process_inner_views(self):
        if False:
            return 10
        for view in self.baseviews:
            for inner_class in view.get_uninit_inner_views():
                for v in self.baseviews:
                    if isinstance(v, inner_class) and v not in view.get_init_inner_views():
                        view.get_init_inner_views().append(v)

def init_appbuilder(app: Flask) -> AirflowAppBuilder:
    if False:
        print('Hello World!')
    'Init `Flask App Builder <https://flask-appbuilder.readthedocs.io/en/latest/>`__.'
    return AirflowAppBuilder(app=app, session=settings.Session, base_template='airflow/main.html', update_perms=conf.getboolean('webserver', 'UPDATE_FAB_PERMS'), auth_rate_limited=conf.getboolean('webserver', 'AUTH_RATE_LIMITED', fallback=True), auth_rate_limit=conf.get('webserver', 'AUTH_RATE_LIMIT', fallback='5 per 40 second'))