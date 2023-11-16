"""RestFull API interface class."""
import os
import sys
import tempfile
from io import open
import webbrowser
import zlib
import socket
from urllib.parse import urljoin
from glances.globals import b, json_dumps
from glances.timer import Timer
from glances.logger import logger
try:
    from bottle import Bottle, static_file, abort, response, request, auth_basic, template, TEMPLATE_PATH
except ImportError:
    logger.critical('Bottle module not found. Glances cannot start in web server mode.')
    sys.exit(2)

def compress(func):
    if False:
        print('Hello World!')
    'Compress result with deflate algorithm if the client ask for it.'

    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        'Wrapper that take one function and return the compressed result.'
        ret = func(*args, **kwargs)
        logger.debug('Receive {} {} request with header: {}'.format(request.method, request.url, ['{}: {}'.format(h, request.headers.get(h)) for h in request.headers.keys()]))
        if 'deflate' in request.headers.get('Accept-Encoding', ''):
            response.headers['Content-Encoding'] = 'deflate'
            ret = deflate_compress(ret)
        else:
            response.headers['Content-Encoding'] = 'identity'
        return ret

    def deflate_compress(data, compress_level=6):
        if False:
            for i in range(10):
                print('nop')
        'Compress given data using the DEFLATE algorithm'
        zobj = zlib.compressobj(compress_level, zlib.DEFLATED, zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, zlib.Z_DEFAULT_STRATEGY)
        return zobj.compress(b(data)) + zobj.flush()
    return wrapper

class GlancesBottle(object):
    """This class manages the Bottle Web server."""
    API_VERSION = '3'

    def __init__(self, config=None, args=None):
        if False:
            for i in range(10):
                print('nop')
        self.config = config
        self.args = args
        self.stats = None
        self.timer = Timer(0)
        self.load_config(config)
        self.bind_url = urljoin('http://{}:{}/'.format(self.args.bind_address, self.args.port), self.url_prefix)
        self._app = Bottle()
        self._app.install(EnableCors())
        if args.password != '':
            self._app.install(auth_basic(self.check_auth))
        self._route()
        self.STATIC_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static/public')
        TEMPLATE_PATH.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static/templates'))

    def load_config(self, config):
        if False:
            print('Hello World!')
        'Load the outputs section of the configuration file.'
        self.url_prefix = '/'
        if config is not None and config.has_section('outputs'):
            n = config.get_value('outputs', 'max_processes_display', default=None)
            logger.debug('Number of processes to display in the WebUI: {}'.format(n))
            self.url_prefix = config.get_value('outputs', 'url_prefix', default='/')
            logger.debug('URL prefix: {}'.format(self.url_prefix))

    def __update__(self):
        if False:
            print('Hello World!')
        if self.timer.finished():
            self.stats.update()
            self.timer = Timer(self.args.cached_time)

    def app(self):
        if False:
            while True:
                i = 10
        return self._app()

    def check_auth(self, username, password):
        if False:
            return 10
        'Check if a username/password combination is valid.'
        if username == self.args.username:
            from glances.password import GlancesPassword
            pwd = GlancesPassword(username=username, config=self.config)
            return pwd.check_password(self.args.password, pwd.get_hash(password))
        else:
            return False

    def _route(self):
        if False:
            return 10
        'Define route.'
        self._app.route('/api/%s/status' % self.API_VERSION, method='GET', callback=self._api_status)
        self._app.route('/api/%s/config' % self.API_VERSION, method='GET', callback=self._api_config)
        self._app.route('/api/%s/config/<item>' % self.API_VERSION, method='GET', callback=self._api_config_item)
        self._app.route('/api/%s/args' % self.API_VERSION, method='GET', callback=self._api_args)
        self._app.route('/api/%s/args/<item>' % self.API_VERSION, method='GET', callback=self._api_args_item)
        self._app.route('/api/%s/help' % self.API_VERSION, method='GET', callback=self._api_help)
        self._app.route('/api/%s/pluginslist' % self.API_VERSION, method='GET', callback=self._api_plugins)
        self._app.route('/api/%s/all' % self.API_VERSION, method='GET', callback=self._api_all)
        self._app.route('/api/%s/all/limits' % self.API_VERSION, method='GET', callback=self._api_all_limits)
        self._app.route('/api/%s/all/views' % self.API_VERSION, method='GET', callback=self._api_all_views)
        self._app.route('/api/%s/<plugin>' % self.API_VERSION, method='GET', callback=self._api)
        self._app.route('/api/%s/<plugin>/history' % self.API_VERSION, method='GET', callback=self._api_history)
        self._app.route('/api/%s/<plugin>/history/<nb:int>' % self.API_VERSION, method='GET', callback=self._api_history)
        self._app.route('/api/%s/<plugin>/top/<nb:int>' % self.API_VERSION, method='GET', callback=self._api_top)
        self._app.route('/api/%s/<plugin>/limits' % self.API_VERSION, method='GET', callback=self._api_limits)
        self._app.route('/api/%s/<plugin>/views' % self.API_VERSION, method='GET', callback=self._api_views)
        self._app.route('/api/%s/<plugin>/<item>' % self.API_VERSION, method='GET', callback=self._api_item)
        self._app.route('/api/%s/<plugin>/<item>/history' % self.API_VERSION, method='GET', callback=self._api_item_history)
        self._app.route('/api/%s/<plugin>/<item>/history/<nb:int>' % self.API_VERSION, method='GET', callback=self._api_item_history)
        self._app.route('/api/%s/<plugin>/<item>/<value>' % self.API_VERSION, method='GET', callback=self._api_value)
        self._app.route('/api/%s/<plugin>/<item>/<value:path>' % self.API_VERSION, method='GET', callback=self._api_value)
        bindmsg = 'Glances RESTful API Server started on {}api/{}'.format(self.bind_url, self.API_VERSION)
        logger.info(bindmsg)
        if not self.args.disable_webui:
            self._app.route('/', method='GET', callback=self._index)
            self._app.route('/<refresh_time:int>', method=['GET'], callback=self._index)
            self._app.route('/<filepath:path>', method='GET', callback=self._resource)
            bindmsg = 'Glances Web User Interface started on {}'.format(self.bind_url)
        else:
            bindmsg = 'The WebUI is disable (--disable-webui)'
        logger.info(bindmsg)
        print(bindmsg)

    def start(self, stats):
        if False:
            return 10
        'Start the bottle.'
        self.stats = stats
        self.plugins_list = self.stats.getPluginsList()
        if self.args.open_web_browser:
            webbrowser.open(self.bind_url, new=2, autoraise=1)
        if self.url_prefix != '/':
            self.main_app = Bottle()
            self.main_app.mount(self.url_prefix, self._app)
            try:
                self.main_app.run(host=self.args.bind_address, port=self.args.port, quiet=not self.args.debug)
            except socket.error as e:
                logger.critical('Error: Can not ran Glances Web server ({})'.format(e))
        else:
            try:
                self._app.run(host=self.args.bind_address, port=self.args.port, quiet=not self.args.debug)
            except socket.error as e:
                logger.critical('Error: Can not ran Glances Web server ({})'.format(e))

    def end(self):
        if False:
            for i in range(10):
                print('nop')
        'End the bottle.'
        logger.info('Close the Web server')
        self._app.close()
        if self.url_prefix != '/':
            self.main_app.close()

    def _index(self, refresh_time=None):
        if False:
            print('Hello World!')
        'Bottle callback for index.html (/) file.'
        if refresh_time is None or refresh_time < 1:
            refresh_time = int(self.args.time)
        self.__update__()
        return template('index.html', refresh_time=refresh_time)

    def _resource(self, filepath):
        if False:
            print('Hello World!')
        'Bottle callback for resources files.'
        return static_file(filepath, root=self.STATIC_PATH)

    @compress
    def _api_status(self):
        if False:
            while True:
                i = 10
        'Glances API RESTful implementation.\n\n        Return a 200 status code.\n        This entry point should be used to check the API health.\n\n        See related issue:  Web server health check endpoint #1988\n        '
        response.status = 200
        return 'Active'

    @compress
    def _api_help(self):
        if False:
            print('Hello World!')
        'Glances API RESTful implementation.\n\n        Return the help data or 404 error.\n        '
        response.content_type = 'application/json; charset=utf-8'
        view_data = self.stats.get_plugin('help').get_view_data()
        try:
            plist = json_dumps(view_data)
        except Exception as e:
            abort(404, 'Cannot get help view data (%s)' % str(e))
        return plist

    @compress
    def _api_plugins(self):
        if False:
            print('Hello World!')
        'Glances API RESTFul implementation.\n\n        @api {get} /api/%s/pluginslist Get plugins list\n        @apiVersion 2.0\n        @apiName pluginslist\n        @apiGroup plugin\n\n        @apiSuccess {String[]} Plugins list.\n\n        @apiSuccessExample Success-Response:\n            HTTP/1.1 200 OK\n            [\n               "load",\n               "help",\n               "ip",\n               "memswap",\n               "processlist",\n               ...\n            ]\n\n         @apiError Cannot get plugin list.\n\n         @apiErrorExample Error-Response:\n            HTTP/1.1 404 Not Found\n        '
        response.content_type = 'application/json; charset=utf-8'
        self.__update__()
        try:
            plist = json_dumps(self.plugins_list)
        except Exception as e:
            abort(404, 'Cannot get plugin list (%s)' % str(e))
        return plist

    @compress
    def _api_all(self):
        if False:
            print('Hello World!')
        'Glances API RESTful implementation.\n\n        Return the JSON representation of all the plugins\n        HTTP/200 if OK\n        HTTP/400 if plugin is not found\n        HTTP/404 if others error\n        '
        response.content_type = 'application/json; charset=utf-8'
        if self.args.debug:
            fname = os.path.join(tempfile.gettempdir(), 'glances-debug.json')
            try:
                with open(fname) as f:
                    return f.read()
            except IOError:
                logger.debug('Debug file (%s) not found' % fname)
        self.__update__()
        try:
            statval = json_dumps(self.stats.getAllAsDict())
        except Exception as e:
            abort(404, 'Cannot get stats (%s)' % str(e))
        return statval

    @compress
    def _api_all_limits(self):
        if False:
            i = 10
            return i + 15
        'Glances API RESTful implementation.\n\n        Return the JSON representation of all the plugins limits\n        HTTP/200 if OK\n        HTTP/400 if plugin is not found\n        HTTP/404 if others error\n        '
        response.content_type = 'application/json; charset=utf-8'
        try:
            limits = json_dumps(self.stats.getAllLimitsAsDict())
        except Exception as e:
            abort(404, 'Cannot get limits (%s)' % str(e))
        return limits

    @compress
    def _api_all_views(self):
        if False:
            print('Hello World!')
        'Glances API RESTful implementation.\n\n        Return the JSON representation of all the plugins views\n        HTTP/200 if OK\n        HTTP/400 if plugin is not found\n        HTTP/404 if others error\n        '
        response.content_type = 'application/json; charset=utf-8'
        try:
            limits = json_dumps(self.stats.getAllViewsAsDict())
        except Exception as e:
            abort(404, 'Cannot get views (%s)' % str(e))
        return limits

    @compress
    def _api(self, plugin):
        if False:
            for i in range(10):
                print('nop')
        'Glances API RESTful implementation.\n\n        Return the JSON representation of a given plugin\n        HTTP/200 if OK\n        HTTP/400 if plugin is not found\n        HTTP/404 if others error\n        '
        response.content_type = 'application/json; charset=utf-8'
        if plugin not in self.plugins_list:
            abort(400, 'Unknown plugin %s (available plugins: %s)' % (plugin, self.plugins_list))
        self.__update__()
        try:
            statval = self.stats.get_plugin(plugin).get_stats()
        except Exception as e:
            abort(404, 'Cannot get plugin %s (%s)' % (plugin, str(e)))
        return statval

    @compress
    def _api_top(self, plugin, nb=0):
        if False:
            return 10
        'Glances API RESTful implementation.\n\n        Return the JSON representation of a given plugin limited to the top nb items.\n        It is used to reduce the payload of the HTTP response (example: processlist).\n\n        HTTP/200 if OK\n        HTTP/400 if plugin is not found\n        HTTP/404 if others error\n        '
        response.content_type = 'application/json; charset=utf-8'
        if plugin not in self.plugins_list:
            abort(400, 'Unknown plugin %s (available plugins: %s)' % (plugin, self.plugins_list))
        self.__update__()
        try:
            statval = self.stats.get_plugin(plugin).get_export()
        except Exception as e:
            abort(404, 'Cannot get plugin %s (%s)' % (plugin, str(e)))
        if isinstance(statval, list):
            return json_dumps(statval[:nb])
        else:
            return json_dumps(statval)

    @compress
    def _api_history(self, plugin, nb=0):
        if False:
            for i in range(10):
                print('nop')
        'Glances API RESTful implementation.\n\n        Return the JSON representation of a given plugin history\n        Limit to the last nb items (all if nb=0)\n        HTTP/200 if OK\n        HTTP/400 if plugin is not found\n        HTTP/404 if others error\n        '
        response.content_type = 'application/json; charset=utf-8'
        if plugin not in self.plugins_list:
            abort(400, 'Unknown plugin %s (available plugins: %s)' % (plugin, self.plugins_list))
        self.__update__()
        try:
            statval = self.stats.get_plugin(plugin).get_stats_history(nb=int(nb))
        except Exception as e:
            abort(404, 'Cannot get plugin history %s (%s)' % (plugin, str(e)))
        return statval

    @compress
    def _api_limits(self, plugin):
        if False:
            print('Hello World!')
        'Glances API RESTful implementation.\n\n        Return the JSON limits of a given plugin\n        HTTP/200 if OK\n        HTTP/400 if plugin is not found\n        HTTP/404 if others error\n        '
        response.content_type = 'application/json; charset=utf-8'
        if plugin not in self.plugins_list:
            abort(400, 'Unknown plugin %s (available plugins: %s)' % (plugin, self.plugins_list))
        try:
            ret = self.stats.get_plugin(plugin).limits
        except Exception as e:
            abort(404, 'Cannot get limits for plugin %s (%s)' % (plugin, str(e)))
        return ret

    @compress
    def _api_views(self, plugin):
        if False:
            i = 10
            return i + 15
        'Glances API RESTful implementation.\n\n        Return the JSON views of a given plugin\n        HTTP/200 if OK\n        HTTP/400 if plugin is not found\n        HTTP/404 if others error\n        '
        response.content_type = 'application/json; charset=utf-8'
        if plugin not in self.plugins_list:
            abort(400, 'Unknown plugin %s (available plugins: %s)' % (plugin, self.plugins_list))
        try:
            ret = self.stats.get_plugin(plugin).get_views()
        except Exception as e:
            abort(404, 'Cannot get views for plugin %s (%s)' % (plugin, str(e)))
        return ret

    def _api_itemvalue(self, plugin, item, value=None, history=False, nb=0):
        if False:
            while True:
                i = 10
        'Father method for _api_item and _api_value.'
        response.content_type = 'application/json; charset=utf-8'
        if plugin not in self.plugins_list:
            abort(400, 'Unknown plugin %s (available plugins: %s)' % (plugin, self.plugins_list))
        self.__update__()
        if value is None:
            if history:
                ret = self.stats.get_plugin(plugin).get_stats_history(item, nb=int(nb))
            else:
                ret = self.stats.get_plugin(plugin).get_stats_item(item)
            if ret is None:
                abort(404, 'Cannot get item %s%s in plugin %s' % (item, 'history ' if history else '', plugin))
        else:
            if history:
                ret = None
            else:
                ret = self.stats.get_plugin(plugin).get_stats_value(item, value)
            if ret is None:
                abort(404, 'Cannot get item %s(%s=%s) in plugin %s' % ('history ' if history else '', item, value, plugin))
        return ret

    @compress
    def _api_item(self, plugin, item):
        if False:
            return 10
        'Glances API RESTful implementation.\n\n        Return the JSON representation of the couple plugin/item\n        HTTP/200 if OK\n        HTTP/400 if plugin is not found\n        HTTP/404 if others error\n\n        '
        return self._api_itemvalue(plugin, item)

    @compress
    def _api_item_history(self, plugin, item, nb=0):
        if False:
            i = 10
            return i + 15
        'Glances API RESTful implementation.\n\n        Return the JSON representation of the couple plugin/history of item\n        HTTP/200 if OK\n        HTTP/400 if plugin is not found\n        HTTP/404 if others error\n\n        '
        return self._api_itemvalue(plugin, item, history=True, nb=int(nb))

    @compress
    def _api_value(self, plugin, item, value):
        if False:
            for i in range(10):
                print('nop')
        'Glances API RESTful implementation.\n\n        Return the process stats (dict) for the given item=value\n        HTTP/200 if OK\n        HTTP/400 if plugin is not found\n        HTTP/404 if others error\n        '
        return self._api_itemvalue(plugin, item, value)

    @compress
    def _api_config(self):
        if False:
            return 10
        'Glances API RESTful implementation.\n\n        Return the JSON representation of the Glances configuration file\n        HTTP/200 if OK\n        HTTP/404 if others error\n        '
        response.content_type = 'application/json; charset=utf-8'
        try:
            args_json = json_dumps(self.config.as_dict())
        except Exception as e:
            abort(404, 'Cannot get config (%s)' % str(e))
        return args_json

    @compress
    def _api_config_item(self, item):
        if False:
            return 10
        'Glances API RESTful implementation.\n\n        Return the JSON representation of the Glances configuration item\n        HTTP/200 if OK\n        HTTP/400 if item is not found\n        HTTP/404 if others error\n        '
        response.content_type = 'application/json; charset=utf-8'
        config_dict = self.config.as_dict()
        if item not in config_dict:
            abort(400, 'Unknown configuration item %s' % item)
        try:
            args_json = json_dumps(config_dict[item])
        except Exception as e:
            abort(404, 'Cannot get config item (%s)' % str(e))
        return args_json

    @compress
    def _api_args(self):
        if False:
            print('Hello World!')
        'Glances API RESTful implementation.\n\n        Return the JSON representation of the Glances command line arguments\n        HTTP/200 if OK\n        HTTP/404 if others error\n        '
        response.content_type = 'application/json; charset=utf-8'
        try:
            args_json = json_dumps(vars(self.args))
        except Exception as e:
            abort(404, 'Cannot get args (%s)' % str(e))
        return args_json

    @compress
    def _api_args_item(self, item):
        if False:
            i = 10
            return i + 15
        'Glances API RESTful implementation.\n\n        Return the JSON representation of the Glances command line arguments item\n        HTTP/200 if OK\n        HTTP/400 if item is not found\n        HTTP/404 if others error\n        '
        response.content_type = 'application/json; charset=utf-8'
        if item not in self.args:
            abort(400, 'Unknown argument item %s' % item)
        try:
            args_json = json_dumps(vars(self.args)[item])
        except Exception as e:
            abort(404, 'Cannot get args item (%s)' % str(e))
        return args_json

class EnableCors(object):
    name = 'enable_cors'
    api = 2

    def apply(self, fn, context):
        if False:
            for i in range(10):
                print('nop')

        def _enable_cors(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'
            if request.method != 'OPTIONS':
                return fn(*args, **kwargs)
        return _enable_cors