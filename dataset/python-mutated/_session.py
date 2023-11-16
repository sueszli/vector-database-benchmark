"""
Definition of the Session class.
"""
import re
import gc
import sys
import time
import json
import base64
import random
import hashlib
import asyncio
import weakref
import datetime
from http.cookies import SimpleCookie
from ..event._component import new_type
from ._component2 import PyComponent, JsComponent, AppComponentMeta
from ._asset import Asset, Bundle, solve_dependencies
from ._assetstore import AssetStore, INDEX
from ._assetstore import assets as assetstore
from ._clientcore import serializer
from . import logger
from .. import config
reprs = json.dumps

def get_random_string(length=24, allowed_chars=None):
    if False:
        return 10
    ' Produce a securely generated random string.\n\n    With a length of 12 with the a-z, A-Z, 0-9 character set returns\n    a 71-bit value. log_2((26+26+10)^12) =~ 71 bits\n    '
    allowed_chars = allowed_chars or 'abcdefghijklmnopqrstuvwxyz' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    try:
        srandom = random.SystemRandom()
    except NotImplementedError:
        srandom = random
        logger.warning('Falling back to less secure Mersenne Twister random string.')
        bogus = '%s%s%s' % (random.getstate(), time.time(), 'sdkhfbsdkfbsdbhf')
        random.seed(hashlib.sha256(bogus.encode()).digest())
    return ''.join((srandom.choice(allowed_chars) for i in range(length)))

class Session:
    """ A connection between Python and the client runtime (JavaScript).

    The session is what holds together the app widget, the web runtime,
    and the websocket instance that connects to it.

    Responsibilities:

    * Send messages to the client and process messages received by the client.
    * Keep track of PyComponent instances used by the session.
    * Keep track of JsComponent instances associated with the session.
    * Ensure that the client has all the module definitions it needs.

    """
    STATUS = new_type('Enum', (), {'PENDING': 1, 'CONNECTED': 2, 'CLOSED': 0})

    def __init__(self, app_name, store=None, request=None):
        if False:
            print('Hello World!')
        self._store = store if store is not None else assetstore
        assert isinstance(self._store, AssetStore)
        self._creation_time = time.time()
        self._id = get_random_string()
        self._app_name = app_name
        self._present_classes = set()
        self._present_modules = set()
        self._present_assets = set()
        self._assets_to_ignore = set()
        self._data = {}
        self._runtime = None
        self._ws = None
        self._closing = False
        self._component = None
        self._component_counter = 0
        self._component_instances = weakref.WeakValueDictionary()
        self._dead_component_ids = set()
        self._ping_calls = []
        self._ping_counter = 0
        self._eval_result = {}
        self._eval_count = 0
        self._pending_commands = []
        self._request = request
        if request and request.cookies:
            cookies = request.cookies
        else:
            cookies = {}
        self._set_cookies(cookies)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        t = '<%s for %r (%i) at 0x%x>'
        return t % (self.__class__.__name__, self.app_name, self.status, id(self))

    @property
    def request(self):
        if False:
            i = 10
            return i + 15
        'The tornado request that was at the origin of this session.\n        '
        return self._request

    @property
    def id(self):
        if False:
            for i in range(10):
                print('nop')
        ' The unique identifier of this session.\n        '
        return self._id

    @property
    def app_name(self):
        if False:
            while True:
                i = 10
        ' The name of the application that this session represents.\n        '
        return self._app_name

    @property
    def app(self):
        if False:
            i = 10
            return i + 15
        ' The root PyComponent or JsComponent instance that represents the app.\n        '
        return self._component

    @property
    def runtime(self):
        if False:
            return 10
        ' The runtime that is rendering this app instance. Can be\n        None if the client is a browser.\n        '
        return self._runtime

    @property
    def status(self):
        if False:
            i = 10
            return i + 15
        ' The status of this session.\n        The lifecycle for each session is:\n\n        * status 1: pending\n        * status 2: connected\n        * status 0: closed\n        '
        if self._ws is None:
            return self.STATUS.PENDING
        elif self._ws.close_code is None:
            return self.STATUS.CONNECTED
        else:
            return self.STATUS.CLOSED

    @property
    def present_modules(self):
        if False:
            return 10
        ' The set of module names that is (currently) available at the client.\n        '
        return set(self._present_modules)

    @property
    def assets_to_ignore(self):
        if False:
            print('Hello World!')
        ' The set of names of assets that should *not* be pushed to\n        the client, e.g. because they are already present on the page.\n        Add names to this set to prevent them from being loaded.\n        '
        return self._assets_to_ignore

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        ' Close the session: close websocket, close runtime, dispose app.\n        '
        self._ping_calls = []
        self._closing = True
        try:
            if self._ws:
                self._ws.close_this()
            if self._runtime:
                self._runtime.close()
            if self._component is not None:
                self._component.dispose()
                self._component = None
            self._data = {}
            gc.collect()
        finally:
            self._closing = False

    def _set_ws(self, ws):
        if False:
            print('Hello World!')
        ' A session is always first created, so we know what page to\n        serve. The client will connect the websocket, and communicate\n        the session_id so it can be connected to the correct Session\n        via this method\n        '
        if self._ws is not None:
            raise RuntimeError('Session is already connected.')
        self._ws = ws
        self._ws.write_command(('PRINT', 'Flexx session says hi'))
        for command in self._pending_commands:
            self._ws.write_command(command)
        self._ws.write_command(('INIT_DONE',))

    def _set_cookies(self, cookies=None):
        if False:
            while True:
                i = 10
        ' To set cookies, must be an http.cookie.SimpleCookie object.\n        When the app is loaded as a web app, the cookies are set *before* the\n        main component is instantiated. Otherwise they are set when the websocket\n        is connected.\n        '
        self._cookies = cookies if cookies else SimpleCookie()

    def _set_runtime(self, runtime):
        if False:
            i = 10
            return i + 15
        if self._runtime is not None:
            raise RuntimeError('Session already has a runtime.')
        self._runtime = runtime

    def get_cookie(self, name, default=None, max_age_days=31, min_version=None):
        if False:
            i = 10
            return i + 15
        ' Gets the value of the cookie with the given name, else default.\n        Note that cookies only really work for web apps.\n        '
        from tornado.web import decode_signed_value
        if name in self._cookies:
            value = self._cookies[name].value
            value = decode_signed_value(config.cookie_secret, name, value, max_age_days=max_age_days, min_version=min_version)
            return value.decode()
        else:
            return default

    def set_cookie(self, name, value, expires_days=30, version=None, domain=None, expires=None, path='/', **kwargs):
        if False:
            while True:
                i = 10
        " Sets the given cookie name/value with the given options. Set value\n        to None to clear. The cookie value is secured using\n        `flexx.config.cookie_secret`; don't forget to set that config\n        value in your server. Additional keyword arguments are set on\n        the Cookie.Morsel directly.\n        "
        from tornado.escape import native_str
        from tornado.httputil import format_timestamp
        from tornado.web import create_signed_value
        if value is None:
            value = ''
            expires = datetime.datetime.utcnow() - datetime.timedelta(days=365)
        else:
            secret = config.cookie_secret
            value = create_signed_value(secret, name, value, version=version, key_version=None)
        name = native_str(name)
        value = native_str(value)
        if re.search('[\\x00-\\x20]', name + value):
            raise ValueError('Invalid cookie %r: %r' % (name, value))
        if name in self._cookies:
            del self._cookies[name]
        self._cookies[name] = value
        morsel = self._cookies[name]
        if domain:
            morsel['domain'] = domain
        if expires_days is not None and (not expires):
            expires = datetime.datetime.utcnow() + datetime.timedelta(days=expires_days)
        if expires:
            morsel['expires'] = format_timestamp(expires)
        if path:
            morsel['path'] = path
        for (k, v) in kwargs.items():
            if k == 'max_age':
                k = 'max-age'
            if k in ['httponly', 'secure'] and (not v):
                continue
            morsel[k] = v
        self.send_command('EXEC', 'document.cookie = "%s";' % morsel.OutputString().replace('"', '\\"'))

    def add_data(self, name, data):
        if False:
            return 10
        " Add data to serve to the client (e.g. images), specific to this\n        session. Returns the link at which the data can be retrieved.\n        Note that actions can be used to send (binary) data directly\n        to the client (over the websocket).\n\n        Parameters:\n            name (str): the name of the data, e.g. 'icon.png'. If data has\n                already been set on this name, it is overwritten.\n            data (bytes): the data blob.\n\n        Returns:\n            str: the (relative) url at which the data can be retrieved.\n        "
        if not isinstance(name, str):
            raise TypeError('Session.add_data() name must be a str.')
        if name in self._data:
            raise ValueError('Session.add_data() got existing name %r.' % name)
        if not isinstance(data, bytes):
            raise TypeError('Session.add_data() data must be bytes.')
        self._data[name] = data
        return 'flexx/data/%s/%s' % (self.id, name)

    def remove_data(self, name):
        if False:
            return 10
        ' Remove the data associated with the given name. If you need this,\n        consider using actions instead. Note that data is automatically\n        released when the session is closed.\n        '
        self._data.pop(name, None)

    def get_data_names(self):
        if False:
            return 10
        ' Get a list of names of the data provided by this session.\n        '
        return list(self._data.keys())

    def get_data(self, name):
        if False:
            while True:
                i = 10
        ' Get the data corresponding to the given name. This can be\n        data local to the session, or global data. Returns None if data\n        by that name is unknown.\n        '
        if True:
            data = self._data.get(name, None)
        if data is None:
            data = self._store.get_data(name)
        return data

    def _dump_data(self):
        if False:
            for i in range(10):
                print('nop')
        ' Get a dictionary that contains all data specific to this session.\n        The keys represent relative paths, the values are all bytes.\n        Private method, used by App.dump().\n        '
        d = {}
        for fname in self.get_data_names():
            d['flexx/data/{}/{}'.format(self.id, fname)] = self.get_data(fname)
        return d

    def _register_component(self, component, id=None):
        if False:
            print('Hello World!')
        ' Called by PyComponent and JsComponent to give them an id\n        and register with the session.\n        '
        assert isinstance(component, (PyComponent, JsComponent))
        assert component.session is self
        cls = component.__class__
        if self._component is None:
            self._component = component
        if id is None:
            self._component_counter += 1
            id = cls.__name__ + '_' + str(self._component_counter)
        component._id = id
        component._uid = self.id + '_' + id
        self._component_instances[component._id] = component
        self._register_component_class(cls)
        self.keep_alive(component)

    def _unregister_component(self, component):
        if False:
            print('Hello World!')
        self._dead_component_ids.add(component.id)

    def get_component_instance(self, id):
        if False:
            for i in range(10):
                print('nop')
        ' Get PyComponent or JsComponent instance that is associated with\n        this session and has the corresponding id. The returned value can be\n        None if it does not exist, and a returned component can be disposed.\n        '
        return self._component_instances.get(id, None)

    def _register_component_class(self, cls):
        if False:
            for i in range(10):
                print('nop')
        ' Mark the given PyComponent or JsComponent class as used; ensure\n        that the client knows about the module that it is defined in,\n        dependencies of this module, and associated assets of any of these\n        modules.\n        '
        if not (isinstance(cls, type) and issubclass(cls, (PyComponent, JsComponent))):
            raise TypeError('_register_component_class() needs a PyComponent or JsComponent class')
        if cls in self._present_classes:
            return
        same_name = [c for c in self._present_classes if c.__name__ == cls.__name__]
        if same_name:
            is_interactive = self._app_name == '__default__'
            same_name.append(cls)
            is_dynamic_cls = all([c.__module__ == '__main__' for c in same_name])
            if not (is_interactive and is_dynamic_cls):
                raise RuntimeError('Cannot have multiple Component classes with the same name unless using interactive session and the classes are dynamically defined: %r' % same_name)
        logger.debug('Registering Component class %r' % cls.__name__)
        self._register_module(cls.__jsmodule__)

    def _register_module(self, mod_name):
        if False:
            print('Hello World!')
        ' Register a module with the client, as well as its\n        dependencies, and associated assets of the module and its\n        dependencies. If the module was already defined, it is\n        re-defined.\n        '
        if mod_name.startswith(('flexx.app', 'flexx.event')) and '.examples' not in mod_name:
            return
        modules = set()
        assets = []

        def collect_module_and_deps(mod):
            if False:
                for i in range(10):
                    print('nop')
            if mod.name.startswith(('flexx.app', 'flexx.event')):
                return
            if mod.name not in self._present_modules:
                self._present_modules.add(mod.name)
                for dep in mod.deps:
                    if dep.startswith(('flexx.app', 'flexx.event')):
                        continue
                    submod = self._store.modules[dep]
                    collect_module_and_deps(submod)
                modules.add(mod)
        self._store.update_modules()
        mod = self._store.modules[mod_name]
        collect_module_and_deps(mod)
        f = lambda m: (m.name.startswith('__main__'), m.name)
        modules = solve_dependencies(sorted(modules, key=f))
        for mod in modules:
            for asset_name in self._store.get_associated_assets(mod.name):
                if asset_name not in self._present_assets:
                    self._present_assets.add(asset_name)
                    assets.append(self._store.get_asset(asset_name))
        if not modules:
            modules.append(mod)
        for mod in modules:
            if mod.get_css().strip():
                assets.append(self._store.get_asset(mod.name + '.css'))
        for mod in modules:
            assets.append(self._store.get_asset(mod.name + '.js'))
        for mod in modules:
            for cls in mod.component_classes:
                self._present_classes.add(cls)
        for asset in assets:
            if asset.name in self._assets_to_ignore:
                continue
            logger.debug('Loading asset %s' % asset.name)
            suffix = asset.name.split('.')[-1].upper()
            if suffix == 'JS' and isinstance(asset, Bundle):
                suffix = 'JS-EVAL'
            self.send_command('DEFINE', suffix, asset.name, asset.to_string())

    def send_command(self, *command):
        if False:
            print('Hello World!')
        ' Send a command to the other side. Commands consists of at least one\n        argument (a string representing the type of command).\n        '
        assert len(command) >= 1
        if self._closing:
            pass
        elif self.status == self.STATUS.CONNECTED:
            self._ws.write_command(command)
        elif self.status == self.STATUS.PENDING:
            self._pending_commands.append(command)
        else:
            logger.warning('Cannot send commands; app is closed')

    def _receive_command(self, command):
        if False:
            for i in range(10):
                print('nop')
        ' Received a command from JS.\n        '
        cmd = command[0]
        if cmd == 'EVALRESULT':
            self._eval_result[command[2]] = command[1]
        elif cmd == 'PRINT':
            print('JS:', command[1])
        elif cmd == 'INFO':
            logger.info('JS: ' + command[1])
        elif cmd == 'WARN':
            logger.warning('JS: ' + command[1])
        elif cmd == 'ERROR':
            logger.error('JS: ' + command[1] + ' - stack trace in browser console (hit F12).')
        elif cmd == 'INVOKE':
            (id, name, args) = command[1:]
            ob = self.get_component_instance(id)
            if ob is None:
                if id not in self._dead_component_ids:
                    t = 'Cannot invoke %s.%s; session does not know it (anymore).'
                    logger.warning(t % (id, name))
            elif ob._disposed:
                pass
            else:
                func = getattr(ob, name, None)
                if func:
                    func(*args)
        elif cmd == 'PONG':
            self._receive_pong(command[1])
        elif cmd == 'INSTANTIATE':
            (modulename, cname, id, args, kwargs) = command[1:]
            c = self.get_component_instance(id)
            if c and (not c._disposed):
                self.keep_alive(c)
                return
            (m, cls, e) = (None, None, 0)
            if modulename in assetstore.modules:
                m = sys.modules[modulename]
                cls = getattr(m, cname, None)
                if cls is None:
                    e = 1
                elif not (isinstance(cls, type) and issubclass(cls, JsComponent)):
                    (cls, e) = (None, 2)
                elif cls not in AppComponentMeta.CLASSES:
                    (cls, e) = (None, 3)
            if cls is None:
                raise RuntimeError('Cannot INSTANTIATE %s.%s (%i)' % (modulename, cname, e))
            kwargs['flx_session'] = self
            kwargs['flx_id'] = id
            assert len(args) == 0
            c = cls(**kwargs)
        elif cmd == 'DISPOSE':
            id = command[1]
            c = self.get_component_instance(id)
            if c and (not c._disposed):
                c._dispose()
            self.send_command('DISPOSE_ACK', command[1])
            self._component_instances.pop(id, None)
        elif cmd == 'DISPOSE_ACK':
            self._component_instances.pop(command[1], None)
            self._dead_component_ids.discard(command[1])
        else:
            logger.error('Unknown command received from JS:\n%s' % command)

    def keep_alive(self, ob, iters=1):
        if False:
            for i in range(10):
                print('nop')
        ' Keep an object alive for a certain amount of time, expressed\n        in Python-JS ping roundtrips. This is intended for making JsComponent\n        (i.e. proxy components) service the time between instantiation\n        triggered from JS and their attachement to a property, though any type\n        of object can be given.\n        '
        ping_to_schedule_at = self._ping_counter + iters
        el = self._get_ping_call_list(ping_to_schedule_at)
        el[1][id(ob)] = ob

    def call_after_roundtrip(self, callback, *args):
        if False:
            print('Hello World!')
        ' A variant of ``call_soon()`` that calls a callback after\n        a py-js roundrip. This can be convenient to delay an action until\n        after other things have settled down.\n        '
        ping_to_schedule_at = self._ping_counter + 1
        el = self._get_ping_call_list(ping_to_schedule_at)
        el.append((callback, args))

    async def co_roundtrip(self):
        """ Coroutine to wait for one Py-JS-Py roundtrip.
        """
        count = 0

        def up():
            if False:
                print('Hello World!')
            nonlocal count
            count += 1
        self.call_after_roundtrip(up)
        while count < 1:
            await asyncio.sleep(0.02)

    async def co_eval(self, js):
        """ Coroutine to evaluate JS in the client, wait for the result,
        and then return it. It is recomended to use this method only
        for testing purposes.
        """
        id = self._eval_count
        self._eval_count += 1
        self.send_command('EVALANDRETURN', js, id)
        while id not in self._eval_result:
            await asyncio.sleep(0.2)
        return self._eval_result.pop(id)

    def _get_ping_call_list(self, ping_count):
        if False:
            i = 10
            return i + 15
        ' Get an element from _ping_call for the specified ping_count.\n        The element is a list [ping_count, {objects}, *(callback, args)]\n        '
        if len(self._ping_calls) == 0:
            send_ping_later(self)
            el = [ping_count, {}]
            self._ping_calls.append(el)
            return el
        for i in reversed(range(len(self._ping_calls))):
            el = self._ping_calls[i]
            if el[0] == ping_count:
                return el
            elif el[0] < ping_count:
                el = [ping_count, {}]
                self._ping_calls.insert(i + 1, el)
                return el
        else:
            el = [ping_count, {}]
            self._ping_calls.insert(0, el)
            return el

    def _receive_pong(self, count):
        if False:
            for i in range(10):
                print('nop')
        while len(self._ping_calls) > 0 and self._ping_calls[0][0] <= count:
            (_, objects, *callbacks) = self._ping_calls.pop(0)
            objects.clear()
            del objects
            for (callback, args) in callbacks:
                asyncio.get_event_loop().call_soon(callback, *args)
        if len(self._ping_calls) > 0:
            send_ping_later(self)

def send_ping_later(session):
    if False:
        for i in range(10):
            print('nop')

    def x(weaksession):
        if False:
            for i in range(10):
                print('nop')
        s = weaksession()
        if s is not None and s.status > 0:
            s._ping_counter += 1
            s.send_command('PING', s._ping_counter)
    asyncio.get_event_loop().call_later(0.01, x, weakref.ref(session))

def get_page(session):
    if False:
        while True:
            i = 10
    " Get the string for the HTML page to render this session's app.\n    Not a lot; all other JS and CSS assets are pushed over the websocket.\n    "
    css_assets = [assetstore.get_asset('reset.css')]
    js_assets = [assetstore.get_asset('flexx-core.js')]
    return _get_page(session, js_assets, css_assets, 3, False)

def get_page_for_export(session, commands, link=0):
    if False:
        return 10
    ' Get the string for an exported HTML page (to run without a server).\n    In this case, there is no websocket to push JS/CSS assets over; these\n    need to be included inside or alongside the main html page.\n    '
    css_assets = [assetstore.get_asset('reset.css')]
    js_assets = [assetstore.get_asset('flexx-core.js')]
    modules = [assetstore.modules[name] for name in session.present_modules]
    f = lambda m: (m.name.startswith('__main__'), m.name)
    modules = solve_dependencies(sorted(modules, key=f))
    asset_names = set()
    for mod in modules:
        for asset_name in assetstore.get_associated_assets(mod.name):
            if asset_name not in asset_names:
                asset_names.add(asset_name)
                asset = assetstore.get_asset(asset_name)
                if asset.name.lower().endswith('.js'):
                    js_assets.append(asset)
                else:
                    css_assets.append(asset)
    for mod in modules:
        if mod.get_css().strip():
            css_assets.append(assetstore.get_asset(mod.name + '.css'))
    for mod in modules:
        js_assets.append(assetstore.get_asset(mod.name + '.js'))
    lines = []
    lines.append('flexx.is_exported = true;\n')
    lines.append('flexx.run_exported_app = function () {')
    lines.append('    var commands_b64 = [')
    for command in commands:
        if command[0] != 'DEFINE':
            command_str = base64.encodebytes(serializer.encode(command)).decode()
            lines.append('        "' + command_str.replace('\n', '') + '",')
    lines.append('        ];')
    lines.append('    bb64 =  flexx.require("bb64");')
    lines.append('    for (var i=0; i<commands_b64.length; i++) {')
    lines.append('        var command = flexx.serializer.decode(bb64.decode(commands_b64[i]));')
    lines.append('        flexx.s1._receive_command(command);')
    lines.append('    }\n};\n')
    export_asset = Asset('flexx-export.js', '\n'.join(lines))
    js_assets.append(export_asset)
    return _get_page(session, js_assets, css_assets, link, True)

def _get_page(session, js_assets, css_assets, link, export):
    if False:
        return 10
    ' Compose index page. Depending on the value of link and the types\n    of assets, the assets are either embedded or linked.\n    '
    pre_path = 'flexx/assets' if export else '/flexx/assets'
    codes = []
    for assets in [css_assets, js_assets]:
        for asset in assets:
            if link in (0, 1):
                html = asset.to_html('{}', link)
            elif asset.name.endswith(('-info.js', '-export.js')):
                html = asset.to_html('', 0)
            else:
                html = asset.to_html(pre_path + '/shared/{}', link)
            codes.append(html)
            if export and assets is js_assets:
                codes.append('<script>window.flexx.spin();</script>')
        codes.append('')
    codes.append('<script>flexx.create_session("%s", "%s");</script>\n' % (session.app_name, session.id))
    src = INDEX
    if link in (0, 1):
        asset_names = [a.name for a in css_assets + js_assets]
        toc = '<!-- Contents:\n\n- ' + '\n- '.join(asset_names) + '\n\n-->'
        codes.insert(0, toc)
        src = src.replace('ASSET-HOOK', '\n\n\n'.join(codes))
    else:
        src = src.replace('ASSET-HOOK', '\n'.join(codes))
    return src