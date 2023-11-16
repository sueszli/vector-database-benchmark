import math
import sys
from flask import abort
from flask import render_template
from flask import request
from peewee import Database
from peewee import DoesNotExist
from peewee import Model
from peewee import Proxy
from peewee import SelectQuery
from playhouse.db_url import connect as db_url_connect

class PaginatedQuery(object):

    def __init__(self, query_or_model, paginate_by, page_var='page', page=None, check_bounds=False):
        if False:
            for i in range(10):
                print('nop')
        self.paginate_by = paginate_by
        self.page_var = page_var
        self.page = page or None
        self.check_bounds = check_bounds
        if isinstance(query_or_model, SelectQuery):
            self.query = query_or_model
            self.model = self.query.model
        else:
            self.model = query_or_model
            self.query = self.model.select()

    def get_page(self):
        if False:
            print('Hello World!')
        if self.page is not None:
            return self.page
        curr_page = request.args.get(self.page_var)
        if curr_page and curr_page.isdigit():
            return max(1, int(curr_page))
        return 1

    def get_page_count(self):
        if False:
            print('Hello World!')
        if not hasattr(self, '_page_count'):
            self._page_count = int(math.ceil(float(self.query.count()) / self.paginate_by))
        return self._page_count

    def get_object_list(self):
        if False:
            while True:
                i = 10
        if self.check_bounds and self.get_page() > self.get_page_count():
            abort(404)
        return self.query.paginate(self.get_page(), self.paginate_by)

    def get_page_range(self, page, total, show=5):
        if False:
            for i in range(10):
                print('nop')
        start = max(page - show // 2, 1)
        stop = min(start + show, total) + 1
        start = max(min(start, stop - show), 1)
        return list(range(start, stop)[:show])

def get_object_or_404(query_or_model, *query):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(query_or_model, SelectQuery):
        query_or_model = query_or_model.select()
    try:
        return query_or_model.where(*query).get()
    except DoesNotExist:
        abort(404)

def object_list(template_name, query, context_variable='object_list', paginate_by=20, page_var='page', page=None, check_bounds=True, **kwargs):
    if False:
        while True:
            i = 10
    paginated_query = PaginatedQuery(query, paginate_by=paginate_by, page_var=page_var, page=page, check_bounds=check_bounds)
    kwargs[context_variable] = paginated_query.get_object_list()
    return render_template(template_name, pagination=paginated_query, page=paginated_query.get_page(), **kwargs)

def get_current_url():
    if False:
        print('Hello World!')
    if not request.query_string:
        return request.path
    return '%s?%s' % (request.path, request.query_string)

def get_next_url(default='/'):
    if False:
        for i in range(10):
            print('nop')
    if request.args.get('next'):
        return request.args['next']
    elif request.form.get('next'):
        return request.form['next']
    return default

class FlaskDB(object):
    """
    Convenience wrapper for configuring a Peewee database for use with a Flask
    application. Provides a base `Model` class and registers handlers to manage
    the database connection during the request/response cycle.

    Usage::

        from flask import Flask
        from peewee import *
        from playhouse.flask_utils import FlaskDB


        # The database can be specified using a database URL, or you can pass a
        # Peewee database instance directly:
        DATABASE = 'postgresql:///my_app'
        DATABASE = PostgresqlDatabase('my_app')

        # If we do not want connection-management on any views, we can specify
        # the view names using FLASKDB_EXCLUDED_ROUTES. The db connection will
        # not be opened/closed automatically when these views are requested:
        FLASKDB_EXCLUDED_ROUTES = ('logout',)

        app = Flask(__name__)
        app.config.from_object(__name__)

        # Now we can configure our FlaskDB:
        flask_db = FlaskDB(app)

        # Or use the "deferred initialization" pattern:
        flask_db = FlaskDB()
        flask_db.init_app(app)

        # The `flask_db` provides a base Model-class for easily binding models
        # to the configured database:
        class User(flask_db.Model):
            email = CharField()

    """

    def __init__(self, app=None, database=None, model_class=Model, excluded_routes=None):
        if False:
            print('Hello World!')
        self.database = None
        self.base_model_class = model_class
        self._app = app
        self._db = database
        self._excluded_routes = excluded_routes or ()
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        if False:
            return 10
        self._app = app
        if self._db is None:
            if 'DATABASE' in app.config:
                initial_db = app.config['DATABASE']
            elif 'DATABASE_URL' in app.config:
                initial_db = app.config['DATABASE_URL']
            else:
                raise ValueError('Missing required configuration data for database: DATABASE or DATABASE_URL.')
        else:
            initial_db = self._db
        if 'FLASKDB_EXCLUDED_ROUTES' in app.config:
            self._excluded_routes = app.config['FLASKDB_EXCLUDED_ROUTES']
        self._load_database(app, initial_db)
        self._register_handlers(app)

    def _load_database(self, app, config_value):
        if False:
            print('Hello World!')
        if isinstance(config_value, Database):
            database = config_value
        elif isinstance(config_value, dict):
            database = self._load_from_config_dict(dict(config_value))
        else:
            database = db_url_connect(config_value)
        if isinstance(self.database, Proxy):
            self.database.initialize(database)
        else:
            self.database = database

    def _load_from_config_dict(self, config_dict):
        if False:
            i = 10
            return i + 15
        try:
            name = config_dict.pop('name')
            engine = config_dict.pop('engine')
        except KeyError:
            raise RuntimeError('DATABASE configuration must specify a `name` and `engine`.')
        if '.' in engine:
            (path, class_name) = engine.rsplit('.', 1)
        else:
            (path, class_name) = ('peewee', engine)
        try:
            __import__(path)
            module = sys.modules[path]
            database_class = getattr(module, class_name)
            assert issubclass(database_class, Database)
        except ImportError:
            raise RuntimeError('Unable to import %s' % engine)
        except AttributeError:
            raise RuntimeError('Database engine not found %s' % engine)
        except AssertionError:
            raise RuntimeError('Database engine not a subclass of peewee.Database: %s' % engine)
        return database_class(name, **config_dict)

    def _register_handlers(self, app):
        if False:
            return 10
        app.before_request(self.connect_db)
        app.teardown_request(self.close_db)

    def get_model_class(self):
        if False:
            return 10
        if self.database is None:
            raise RuntimeError('Database must be initialized.')

        class BaseModel(self.base_model_class):

            class Meta:
                database = self.database
        return BaseModel

    @property
    def Model(self):
        if False:
            return 10
        if self._app is None:
            database = getattr(self, 'database', None)
            if database is None:
                self.database = Proxy()
        if not hasattr(self, '_model_class'):
            self._model_class = self.get_model_class()
        return self._model_class

    def connect_db(self):
        if False:
            print('Hello World!')
        if self._excluded_routes and request.endpoint in self._excluded_routes:
            return
        self.database.connect()

    def close_db(self, exc):
        if False:
            i = 10
            return i + 15
        if self._excluded_routes and request.endpoint in self._excluded_routes:
            return
        if not self.database.is_closed():
            self.database.close()