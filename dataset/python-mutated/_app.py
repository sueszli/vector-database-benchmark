"""
Definition of the App class and app manager.
"""
import io
import os
import sys
import time
import weakref
import zipfile
from base64 import encodebytes
import webruntime
from .. import config, event
from ._component2 import PyComponent, JsComponent
from ._server import current_server
from ._session import Session, get_page_for_export
from ._assetstore import assets
from . import logger

class ExporterWebSocketDummy:
    """ Object that can be used by an app inplace of the websocket to
    export apps to standalone HTML. The object tracks the commands send
    by the app, so that these can be re-played in the exported document.
    """
    close_code = None

    def __init__(self):
        if False:
            print('Hello World!')
        self.commands = []

    def write_command(self, cmd):
        if False:
            while True:
                i = 10
        self.commands.append(cmd)

class App:
    """ Specification of a Flexx app.

    Strictly speaking, this is a container for a ``PyComponent``/``JsComponent``
    class plus the args and kwargs that it is to be instantiated with.

    Arguments:
        cls (Component): the PyComponent or JsComponent class (e.g. Widget) that
            represents this app.
        args: positional arguments used to instantiate the class (and received
            in its ``init()`` method).
        kwargs: keyword arguments used to initialize the component's properties.
    """

    def __init__(self, cls, *args, **kwargs):
        if False:
            print('Hello World!')
        if not isinstance(cls, type) and issubclass(type, (PyComponent, JsComponent)):
            raise ValueError('App needs a PyComponent or JsComponent class as its first argument.')
        self._cls = cls
        self.args = args
        self.kwargs = kwargs
        self._path = cls.__name__
        self._is_served = False
        if hasattr(cls, 'title') and self.kwargs.get('title', None) is None:
            self.kwargs['title'] = 'Flexx app - ' + cls.__name__
        if hasattr(cls, 'set_icon') and self.kwargs.get('icon', None) is None:
            fname = os.path.abspath(os.path.join(__file__, '..', '..', 'resources', 'flexx.ico'))
            icon_str = encodebytes(open(fname, 'rb').read()).decode()
            self.kwargs['icon'] = 'data:image/ico;base64,' + icon_str

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        a = list(self.args) + list(args)
        kw = {}
        kw.update(self.kwargs)
        kw.update(kwargs)
        return self.cls(*a, **kw)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        t = '<App based on class %s pre-initialized with %i args and %i kwargs>'
        return t % (self.cls.__name__, len(self.args), len(self.kwargs))

    @property
    def cls(self):
        if False:
            i = 10
            return i + 15
        ' The Component class that is the basis of this app.\n        '
        return self._cls

    @property
    def is_served(self):
        if False:
            return 10
        ' Whether this app is already registered by the app manager.\n        '
        return self._is_served

    @property
    def url(self):
        if False:
            print('Hello World!')
        " The url to acces this app. This raises an error if serve() has not\n        been called yet or if Flexx' server is not yet running.\n        "
        server = current_server(False)
        if not self._is_served:
            raise RuntimeError('Cannot determine app url if app is not yet "served".')
        elif not (server and server.serving):
            raise RuntimeError('Cannot determine app url if the server is not yet running.')
        else:
            proto = server.protocol
            (host, port) = server.serving
            path = self._path + '/' if self._path else ''
            return '%s://%s:%i/%s' % (proto, host, port, path)

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        ' The name of the app, i.e. the url path that this app is served at.\n        '
        return self._path or '__main__'

    def serve(self, name=None):
        if False:
            print('Hello World!')
        " Start serving this app.\n\n        This registers the given class with the internal app manager. The\n        app can be loaded via 'http://hostname:port/name'.\n\n        Arguments:\n            name (str, optional): the relative URL path to serve the app on.\n                If this is ``''`` (the empty string), this will be the main app.\n                If not given or None, the name of the component class is used.\n        "
        if self._is_served:
            raise RuntimeError('This app (%s) is already served.' % self.name)
        if name is not None:
            self._path = name
        manager.register_app(self)
        self._is_served = True

    def launch(self, runtime=None, **runtime_kwargs):
        if False:
            i = 10
            return i + 15
        " Launch this app as a desktop app in the given runtime.\n        See https://webruntime.readthedocs.io for details.\n\n        Arguments:\n            runtime (str): the runtime to launch the application in.\n                Default 'app or browser'.\n            runtime_kwargs: kwargs to pass to the ``webruntime.launch`` function\n                and the create_server function.\n                create_server takes 'host', 'port', 'loop', 'backend' as parameters.\n                For webruntime.launch, a few names are passed to runtime kwargs if not \n                already present ('title' and 'icon').\n\n        Returns:\n            Component: an instance of the given class.\n        "
        server_kwargs = {}
        server_keys = ['host', 'port', 'loop', 'backend']
        for (key, value) in runtime_kwargs.items():
            if key in server_keys:
                server_kwargs[key] = value
        current_server(**server_kwargs)
        for key in server_keys:
            if key in runtime_kwargs:
                del runtime_kwargs[key]
        if not self._is_served:
            self.serve()
        session = manager.create_session(self.name)
        if runtime_kwargs.get('title', None) is None and 'title' in self.kwargs:
            runtime_kwargs['title'] = self.kwargs['title']
        if runtime_kwargs.get('icon', None) is None and 'icon' in self.kwargs:
            runtime_kwargs['icon'] = self.kwargs['icon']
        url = self.url + '?session_id=%s' % session.id
        if not runtime or '!' in config.webruntime:
            runtime = config.webruntime.strip('!')
        session._runtime = webruntime.launch(url, runtime=runtime, **runtime_kwargs)
        return session.app

    def dump(self, fname=None, link=2):
        if False:
            return 10
        ' Get a dictionary of web assets that statically represents the app.\n\n        The returned dict contains at least one html file. Any\n        session-specific or shared data is also included. If link is\n        2/3, all shared assets are included too (and the main document\n        links to them). A link value of 0/1 may be prefered for\n        performance or ease of distribution, but with link 2/3 debugging\n        is easier and multiple apps can share common assets.\n\n        When a process only dumps/exports an app, no server is started.\n        Tornado is not even imported (we have a test for this). This makes\n        it possible to use Flexx to dump an app and then serve it with any\n        tool one likes.\n\n        Arguments:\n            fname (str, optional): the name of the main html asset.\n                If not given or None, the name of the component class\n                is used. Must end in .html/.htm/.hta.\n            link (int): whether to link (JS and CSS) assets or embed them:\n                A values of 0/1 is recommended for single (and standalone) apps,\n                while multiple apps can share common assets by using 2/3.\n\n                * 0: all assets are embedded into the main html document.\n                * 1: normal assets are embedded, remote assets remain remote.\n                * 2: all assets are linked (as separate files). Default.\n                * 3: normal assets are linked, remote assets remain remote.\n\n        Returns:\n            dict: A collection of assets.\n        '
        if fname is None:
            if self.name in ('__main__', ''):
                fname = 'index.html'
            else:
                fname = self.name.lower() + '.html'
        if os.path.basename(fname) != fname:
            raise ValueError('App.dump() fname must not contain directory names.')
        elif not fname.lower().endswith(('.html', 'htm', '.hta')):
            raise ValueError('Invalid extension for dumping {}'.format(fname))
        name = fname.split('.')[0].replace('-', '_').replace(' ', '_')
        session = Session(name)
        session._id = name
        self(flx_session=session, flx_is_app=True)
        exporter = ExporterWebSocketDummy()
        session._set_ws(exporter)
        assert link in (0, 1, 2, 3), 'Expecting link to be in (0, 1, 2, 3).'
        if issubclass(self.cls, PyComponent):
            logger.warning('Exporting a PyComponent - any Python interactivity will not work in exported apps.')
        d = {}
        html = get_page_for_export(session, exporter.commands, link)
        if fname.lower().endswith('.hta'):
            hta_tag = '<meta http-equiv="x-ua-compatible" content="ie=edge" />'
            html = html.replace('<head>', '<head>\n    ' + hta_tag, 1)
        d[fname] = html.encode()
        if link in (2, 3):
            d.update(assets._dump_assets(link == 2))
        d.update(session._dump_data())
        d.update(assets._dump_data())
        return d

    def export(self, filename, link=2, overwrite=True):
        if False:
            return 10
        ' Export this app to a static website.\n\n        Also see dump(). An app that contains no data, can be exported to a\n        single html document by setting link to 0.\n\n        Arguments:\n            filename (str): Path to write the HTML document to.\n                If the filename ends with .hta, a Windows HTML Application is\n                created. If a directory is given, the app is exported to\n                appname.html in that directory.\n            link (int): whether to link (JS and CSS) assets or embed them:\n\n                * 0: all assets are embedded into the main html document.\n                * 1: normal assets are embedded, remote assets remain remote.\n                * 2: all assets are linked (as separate files). Default.\n                * 3: normal assets are linked, remote assets remain remote.\n            overwrite (bool, optional): if True (default) will overwrite files\n                that already exist. Otherwise existing files are skipped.\n                The latter makes it possible to efficiently export a series of\n                apps to the same directory and have them share common assets.\n        '
        if not isinstance(filename, str):
            raise ValueError('str filename required, use dump() for in-memory export.')
        filename = os.path.abspath(os.path.expanduser(filename))
        if os.path.isdir(filename) or filename.endswith(('/', '\\')) or '.' not in os.path.basename(filename):
            dirname = filename
            fname = None
        else:
            (dirname, fname) = os.path.split(filename)
        d = self.dump(fname, link)
        for (fname, blob) in d.items():
            filename2 = os.path.join(dirname, fname)
            if not overwrite and os.path.isfile(filename2):
                continue
            dname = os.path.dirname(filename2)
            if not os.path.isdir(dname):
                os.makedirs(dname)
            with open(filename2, 'wb') as f:
                f.write(blob)
        app_type = 'standalone app' if len(d) == 1 else 'app'
        logger.info('Exported %s to %r' % (app_type, filename))

    def publish(self, name, token, url=None):
        if False:
            for i in range(10):
                print('nop')
        ' Publish this app as static HTML on the web.\n\n        This is an experimental feature! We will try to keep your app published,\n        but make no guarantees. We reserve the right to remove apps or shut down\n        the web server completely.\n\n        Arguments:\n            name (str): The name by which to publish this app. Must be unique\n                within the scope of the published site.\n            token (str): a secret token. This is stored at the target website.\n                Subsequent publications of the same app name must have the same\n                token.\n            url (str): The url to POST the app to. If None (default),\n                the default Flexx live website url will be used.\n        '
        d = self.dump('index.html', 2)
        f = io.BytesIO()
        with zipfile.ZipFile(f, 'w') as zf:
            for fname in d.keys():
                zf.writestr(fname, d[fname])
        try:
            import requests
        except ImportError:
            raise ImportError('App.publish() needs requests lib: pip install requests')
        url = url or 'http://flexx.app/submit/{name}/{token}'
        real_url = url.format(name=name, token=token)
        r = requests.post(real_url, data=f.getvalue())
        if r.status_code != 200:
            raise RuntimeError('Publish failed: ' + r.text)
        else:
            print('Publish succeeded, ' + r.text)
            if url.startswith('http://flexx.app'):
                print('You app is now available at http://flexx.app/open/%s/' % name)

    def freeze(self, dirname, launch='firefox-app', excludes=('numpy',), includes=()):
        if False:
            while True:
                i = 10
        ' Create an executable that can be distributed as a standalone\n        desktop application. This process (known as "freezing") requires PyInstaller.\n\n        Note: this method is experimental. See\n        https://flexx.readthedocs.io/en/stable/freeze.html for more information.\n\n        Arguments:\n            dirname (str): Path to generate the executable in. Some temporary\n                files and directories will be created during the freezing process.\n                The actual executable will be placed in "dist/app_name", where\n                app_name is the (lowercase) name of the application class.\n            launch (str): The argument to use for the call to ``flx.launch()``.\n                If set to None, ``flx.serve()`` will be used instead, and you\n                will be responsible for connecting a browser.\n            excludes (list): A list of module names to exclude during freezing.\n                By default Numpy is excluded because PyInstaller detects it\n                even though Flexx does not use it. Override this if you do use Numpy.\n            includes (list): A list of module name to include during freezing.\n\n        '
        from flexx.util import freeze
        import PyInstaller.__main__
        name = self._cls.__name__.lower()
        if dirname.startswith('~'):
            dirname = os.path.expanduser(dirname)
        else:
            dirname = os.path.abspath(dirname)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        main_module = self._cls.__module__
        if main_module == '__main__':
            main_module = 'mainmain'
            main_code = open(sys.modules['__main__'].__file__, 'rb').read().decode()
            for line in main_code.splitlines():
                if '.freeze(' in line or '.run(' in line:
                    comment = ' ' * (len(line) - len(line.lstrip())) + 'pass  # '
                    main_code = main_code.replace(line, comment + line.lstrip())
        script_filename = os.path.join(dirname, name + '.py')
        lines = ['from flexx.util import freeze', 'freeze.install()']
        lines.append('from {} import {} as Main'.format(main_module, self._cls.__name__))
        lines.append('from flexx import flx')
        args_str = ', '.join((repr(x) for x in self.args))
        kwargs_str = ', '.join((key + '=' + repr(x) for (key, x) in self.kwargs.items()))
        args_str = ', ' + args_str if args_str else args_str
        kwargs_str = ', ' + kwargs_str if kwargs_str else kwargs_str
        lines.append('app = flx.App(Main{}{})'.format(args_str, kwargs_str))
        if launch:
            lines.append("app.launch('{}')".format(launch))
        else:
            lines.append("app.serve('')")
        lines.append('flx.run()')
        with open(script_filename, 'wb') as f:
            f.write('\n'.join(lines).encode())
        distdir = 'dist'
        cmd = ['--clean', '--onedir', '--name', name, '--distpath', distdir]
        if sys.platform.startswith('win'):
            cmd.append('--windowed')
        elif sys.platform.startswith('darwin'):
            cmd.append('--windowed')
        for module_name in excludes:
            cmd.extend(['--exclude-module', module_name])
        for module_name in includes:
            cmd.extend(['--include-module', module_name])
        cmd.append(script_filename)
        ori_dir = os.getcwd()
        os.chdir(dirname)
        try:
            PyInstaller.__main__.run(cmd)
        finally:
            os.chdir(ori_dir)
        appdir = os.path.join(dirname, distdir, name)
        assets.update_modules()
        for module_name in {x.split('.')[0] for x in assets.modules.keys()}:
            if module_name == '__main__':
                fname = os.path.join(appdir, 'source', main_module + '.py')
                with open(fname, 'wb') as f:
                    f.write(main_code.encode())
            else:
                freeze.copy_module(module_name, appdir)

def valid_app_name(name):
    if False:
        print('Hello World!')
    T = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'
    return name and name[0] in T[:-10] and all([c in T for c in name])

class AppManager(event.Component):
    """ Manage apps, or more specifically, the session objects.

    There is one AppManager class (in ``flexx.app.manager``). It's
    purpose is to manage the application classes and instances. It is mostly
    intended for internal use, but users can use it to e.g. monitor connections.
    Create a reaction using ``@app.manager.reaction('connections_changed')``
    to track when the number of connected session changes.
    """
    total_sessions = 0

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._appinfo = {}
        self._session_map = weakref.WeakValueDictionary()
        self._last_check_time = time.time()

    def register_app(self, app):
        if False:
            return 10
        ' Register an app (an object that wraps a Component class plus init args).\n        After registering an app (and starting the server) it is\n        possible to connect to "http://address:port/app_name".\n        '
        assert isinstance(app, App)
        name = app.name
        if not valid_app_name(name):
            raise ValueError('Given app does not have a valid name %r' % name)
        (pending, connected) = ([], [])
        if name in self._appinfo:
            (old_app, pending, connected) = self._appinfo[name]
            if app.cls is not old_app.cls:
                logger.warning('Re-defining app class %r' % name)
        self._appinfo[name] = (app, pending, connected)

    def create_default_session(self, cls=None):
        if False:
            print('Hello World!')
        ' Create a default session for interactive use (e.g. the notebook).\n        '
        if '__default__' in self._appinfo:
            raise RuntimeError('The default session can only be created once.')
        if cls is None:
            cls = JsComponent
        if not isinstance(cls, type) and issubclass(cls, (PyComponent, JsComponent)):
            raise TypeError('create_default_session() needs a JsComponent subclass.')
        app = App(cls)
        app.serve('__default__')
        session = Session('__default__')
        self._session_map[session.id] = session
        (_, pending, connected) = self._appinfo['__default__']
        pending.append(session)
        app(flx_session=session, flx_is_app=True)
        return session

    def remove_default_session(self):
        if False:
            i = 10
            return i + 15
        ' Remove default session if there is one, closing the session.\n        '
        s = self.get_default_session()
        if s is not None:
            s.close()
        self._appinfo.pop('__default__', None)

    def get_default_session(self):
        if False:
            return 10
        " Get the default session that is used for interactive use.\n        Returns None unless create_default_session() was called earlier.\n\n        When a JsComponent class is created without a session, this method\n        is called to get one (and will then fail if it's None).\n        "
        x = self._appinfo.get('__default__', None)
        if x is None:
            return None
        else:
            (_, pending, connected) = x
            sessions = pending + connected
            if sessions:
                return sessions[-1]

    def _clear_old_pending_sessions(self, max_age=30):
        if False:
            i = 10
            return i + 15
        try:
            count = 0
            for name in self._appinfo:
                if name == '__default__':
                    continue
                (_, pending, _) = self._appinfo[name]
                to_remove = [s for s in pending if time.time() - s._creation_time > max_age]
                for s in to_remove:
                    self._session_map.pop(s.id, None)
                    pending.remove(s)
                count += len(to_remove)
            if count:
                logger.warning('Cleared %i old pending sessions' % count)
        except Exception as err:
            logger.error('Error when clearing old pending sessions: %s' % str(err))

    def create_session(self, name, id=None, request=None):
        if False:
            for i in range(10):
                print('nop')
        ' Create a session for the app with the given name.\n\n        Instantiate an app and matching session object corresponding\n        to the given name, and return the session. The client should\n        be connected later via connect_client().\n        '
        if time.time() - self._last_check_time > 5:
            self._last_check_time = time.time()
            self._clear_old_pending_sessions()
        if name == '__default__':
            raise RuntimeError('There can be only one __default__ session.')
        elif name not in self._appinfo:
            raise ValueError('Can only instantiate a session with a valid app name.')
        (app, pending, connected) = self._appinfo[name]
        session = Session(name, request=request)
        if id is not None:
            session._id = id
        self._session_map[session.id] = session
        app(flx_session=session, flx_is_app=True)
        pending.append(session)
        logger.debug('Instantiate app client %s' % session.app_name)
        return session

    def connect_client(self, ws, name, session_id, cookies=None):
        if False:
            return 10
        ' Connect a client to a session that was previously created.\n        '
        (_, pending, connected) = self._appinfo[name]
        for session in pending:
            if session.id == session_id:
                pending.remove(session)
                break
        else:
            raise RuntimeError('Asked for session id %r, but could not find it' % session_id)
        assert session.id == session_id
        assert session.status == Session.STATUS.PENDING
        logger.info('New session %s %s' % (name, session_id))
        session._set_cookies(cookies)
        session._set_ws(ws)
        connected.append(session)
        AppManager.total_sessions += 1
        self.connections_changed(session.app_name)
        return session

    def disconnect_client(self, session):
        if False:
            return 10
        ' Close a connection to a client.\n\n        This is called by the websocket when the connection is closed.\n        The manager will remove the session from the list of connected\n        instances.\n        '
        if session.app_name == '__default__':
            logger.info('Default session lost connection to client.')
            return
        (_, pending, connected) = self._appinfo[session.app_name]
        try:
            connected.remove(session)
        except ValueError:
            pass
        logger.info('Session closed %s %s' % (session.app_name, session.id))
        session.close()
        self.connections_changed(session.app_name)

    def has_app_name(self, name):
        if False:
            for i in range(10):
                print('nop')
        ' Returns the case-corrected name if the given name matches\n        a registered appliciation (case insensitive). Returns None if the\n        given name does not match any applications.\n        '
        name = name.lower()
        for key in self._appinfo.keys():
            if key.lower() == name:
                return key
        else:
            return None

    def get_app_names(self):
        if False:
            for i in range(10):
                print('nop')
        ' Get a list of registered application names.\n        '
        return [name for name in sorted(self._appinfo.keys())]

    def get_session_by_id(self, id):
        if False:
            print('Hello World!')
        ' Get session object by its id, or None.\n        '
        return self._session_map.get(id, None)

    def get_connections(self, name):
        if False:
            print('Hello World!')
        ' Given an app name, return the connected session objects.\n        '
        (_, pending, connected) = self._appinfo[name]
        return list(connected)

    @event.emitter
    def connections_changed(self, name):
        if False:
            return 10
        ' Emits an event with the name of the app for which a\n        connection is added or removed.\n        '
        return dict(name=str(name))
manager = AppManager()