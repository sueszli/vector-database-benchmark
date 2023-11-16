"""
The client's core Flexx engine, implemented in PScript.
"""
from pscript import this_is_js, RawJS
from pscript.stubs import window, undefined, time, console, JSON
__pscript__ = True

class Flexx:
    """ JavaScript Flexx module. This provides the connection between
    the Python and JS (via a websocket).
    """

    def __init__(self):
        if False:
            print('Hello World!')
        if window.flexx.init:
            raise RuntimeError('Should not create global Flexx object more than once.')
        self.is_notebook = False
        self.is_exported = False
        for key in window.flexx.keys():
            self[key] = window.flexx[key]
        self.need_main_widget = True
        self._session_count = 0
        self.sessions = {}
        window.addEventListener('load', self.init, False)
        window.addEventListener('unload', self.exit, False)

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        ' Called after document is loaded. '
        self.asset_node = window.document.createElement('div')
        self.asset_node.id = 'Flexx asset container'
        window.document.body.appendChild(self.asset_node)
        if self.is_exported:
            if self.is_notebook:
                print('Flexx: I am in an exported notebook!')
            else:
                print('Flexx: I am in an exported app!')
                self.run_exported_app()
        else:
            print('Flexx: Initializing')
            if not self.is_notebook:
                self._remove_querystring()
            self.init_logging()

    def _remove_querystring(self):
        if False:
            print('Hello World!')
        try:
            window.history.replaceState(window.history.state, '', window.location.pathname)
        except Exception:
            pass

    def exit(self):
        if False:
            i = 10
            return i + 15
        ' Called when runtime is about to quit. '
        for session in self.sessions.values():
            session.exit()

    def spin(self, n=1):
        if False:
            return 10
        RawJS("\n        var el = window.document.getElementById('flexx-spinner');\n        if (el) {\n            if (n === null) {  // Hide the spinner overlay, now or in a bit\n                if (el.children[0].innerHTML.indexOf('limited') > 0) {\n                    setTimeout(function() { el.style.display = 'none'; }, 2000);\n                } else {\n                    el.style.display = 'none';\n                }\n            } else {\n                for (var i=0; i<n; i++) { el.children[1].innerHTML += '&#9632'; }\n            }\n        }\n        ")

    def init_logging(self):
        if False:
            for i in range(10):
                print('nop')
        ' Setup logging so that messages are proxied to Python.\n        '
        if window.console.ori_log:
            return
        window.console.ori_log = window.console.log
        window.console.ori_info = window.console.info or window.console.log
        window.console.ori_warn = window.console.warn or window.console.log
        window.console.ori_error = window.console.error or window.console.log

        def log(msg):
            if False:
                for i in range(10):
                    print('nop')
            window.console.ori_log(msg)
            for session in self.sessions.values():
                session.send_command('PRINT', str(msg))

        def info(msg):
            if False:
                for i in range(10):
                    print('nop')
            window.console.ori_info(msg)
            for session in self.sessions.values():
                session.send_command('INFO', str(msg))

        def warn(msg):
            if False:
                print('Hello World!')
            window.console.ori_warn(msg)
            for session in self.sessions.values():
                session.send_command('WARN', str(msg))

        def error(msg):
            if False:
                print('Hello World!')
            evt = dict(message=str(msg), error=msg, preventDefault=lambda : None)
            on_error(evt)

        def on_error(evt):
            if False:
                for i in range(10):
                    print('nop')
            self._handle_error(evt)
        on_error = on_error.bind(self)
        window.console.log = log
        window.console.info = info
        window.console.warn = warn
        window.console.error = error
        window.addEventListener('error', on_error, False)

    def create_session(self, app_name, session_id, ws_url):
        if False:
            return 10
        if window.performance and window.performance.navigation.type == 2:
            window.location.reload()
        elif self._validate_browser_capabilities():
            s = JsSession(app_name, session_id, ws_url)
            self._session_count += 1
            self['s' + self._session_count] = s
            self.sessions[session_id] = s

    def _validate_browser_capabilities(self):
        if False:
            i = 10
            return i + 15
        RawJS("\n        var el = window.document.getElementById('flexx-spinner');\n        if (    window.WebSocket === undefined || // IE10+\n                Object.keys === undefined || // IE9+\n                false\n           ) {\n            var msg = ('Flexx does not support this browser.<br>' +\n                       'Try Firefox, Chrome, ' +\n                       'or a more recent version of the current browser.');\n            if (el) { el.children[0].innerHTML = msg; }\n            else { window.alert(msg); }\n            return false;\n        } else if (''.startsWith === undefined) { // probably IE\n            var msg = ('Flexx support for this browser is limited.<br>' +\n                       'Consider using Firefox, Chrome, or maybe Edge.');\n            if (el) { el.children[0].innerHTML = msg; }\n            return true;\n        } else {\n            return true;\n        }\n        ")

    def _handle_error(self, evt):
        if False:
            print('Hello World!')
        msg = short_msg = evt.message
        if not window.evt:
            window.evt = evt
        if evt.error and evt.error.stack:
            stack = evt.error.stack.splitlines()
            session_needle = '?session_id=' + self.id
            for i in range(len(stack)):
                stack[i] = stack[i].replace('@', ' @ ').replace(session_needle, '')
            for x in [evt.message, '_pyfunc_op_error']:
                if x in stack[0]:
                    stack.pop(0)
            for i in range(len(stack)):
                for x in ['_process_actions', '_process_reactions', '_process_calls']:
                    if 'Loop.' + x in stack[i]:
                        stack = stack[:i]
                        break
            for i in reversed(range(len(stack))):
                for x in ['flx_action ']:
                    if stack[i] and stack[i].count(x):
                        stack.pop(i)
            msg += '\n' + '\n'.join(stack)
        elif evt.message and evt.lineno:
            msg += '\nIn %s:%i' % (evt.filename, evt.lineno)
        evt.preventDefault()
        window.console.ori_error(msg)
        for session in self.sessions.values():
            session.send_command('ERROR', short_msg)

class JsSession:

    def __init__(self, app_name, id, ws_url=None):
        if False:
            i = 10
            return i + 15
        self.app = None
        self.app_name = app_name
        self.id = id
        self.status = 1
        self.ws_url = ws_url
        self._component_counter = 0
        self._disposed_ob = {'_disposed': True}
        if not self.id:
            jconfig = window.document.getElementById('jupyter-config-data')
            if jconfig:
                try:
                    config = JSON.parse(jconfig.innerText)
                    self.id = config.flexx_session_id
                    self.app_name = config.flexx_app_name
                except Exception as err:
                    print(err)
        self._init_time = time()
        self._pending_commands = []
        self._asset_count = 0
        self._ws = None
        self.last_msg = None
        self.instances = {}
        self.instances_to_check_size = {}
        if not window.flexx.is_exported:
            self.init_socket()
        window.addEventListener('resize', self._check_size_of_objects, False)
        window.setInterval(self._check_size_of_objects, 1000)

    def exit(self):
        if False:
            return 10
        if self._ws:
            self._ws.close()
            self._ws = None
            self.status = 0

    def send_command(self, *command):
        if False:
            while True:
                i = 10
        if self._ws is not None:
            try:
                bb = serializer.encode(command)
            except Exception as err:
                print('Command that failed to encode:')
                print(command)
                raise err
            self._ws.send(bb)

    def instantiate_component(self, module, cname, id, args, kwargs, active_components):
        if False:
            print('Hello World!')
        c = self.instances.get(id, None)
        if c is not None and c._disposed is False:
            return c
        m = window.flexx.require(module)
        Cls = m[cname]
        kwargs['flx_session'] = self
        kwargs['flx_id'] = id
        active_components = active_components or []
        for ac in active_components:
            ac.__enter__()
        try:
            c = Cls(*args, **kwargs)
        finally:
            for ac in reversed(active_components):
                ac.__exit__()
        return c

    def _register_component(self, c, id=None):
        if False:
            i = 10
            return i + 15
        if self.app is None:
            self.app = c
        if id is None:
            self._component_counter += 1
            id = c.__name__ + '_' + str(self._component_counter) + 'js'
        c._id = id
        c._uid = self.id + '_' + id
        self.instances[c._id] = c

    def _unregister_component(self, c):
        if False:
            return 10
        self.instances_to_check_size.pop(c.id, None)
        pass

    def get_component_instance(self, id):
        if False:
            i = 10
            return i + 15
        ' Get instance of a Component class, or None. Or the document body\n        if "body" is given.\n        '
        if id == 'body':
            return window.document.body
        else:
            return self.instances.get(id, None)

    def init_socket(self):
        if False:
            print('Hello World!')
        ' Make the connection to Python.\n        '
        WebSocket = window.WebSocket
        if WebSocket is undefined:
            window.document.body.textContent = 'Browser does not support WebSockets'
            raise RuntimeError('FAIL: need websocket')
        if not self.ws_url:
            proto = 'ws'
            if window.location.protocol == 'https:':
                proto = 'wss'
            address = window.location.hostname
            if window.location.port:
                address += ':' + window.location.port
            self.ws_url = '%s://%s/flexx/ws/%s' % (proto, address, self.app_name)
        self.ws_url = self.ws_url.replace('0.0.0.0', window.location.hostname)
        self._ws = ws = WebSocket(self.ws_url)
        ws.binaryType = 'arraybuffer'
        self.status = 2

        def on_ws_open(evt):
            if False:
                i = 10
                return i + 15
            window.console.info('Socket opened with session id ' + self.id)
            self.send_command('HI_FLEXX', self.id)

        def on_ws_message(evt):
            if False:
                for i in range(10):
                    print('nop')
            msg = evt.data
            if not msg:
                pass
            elif self._pending_commands is None:
                self._receive_raw_command(msg)
            else:
                if len(self._pending_commands) == 0:
                    window.setTimeout(self._process_commands, 0)
                self._pending_commands.push(msg)

        def on_ws_close(evt):
            if False:
                print('Hello World!')
            self._ws = None
            self.status = 0
            msg = 'Lost connection with server'
            if evt and evt.reason:
                msg += ': %s (%i)' % (evt.reason, evt.code)
            if not window.flexx.is_notebook:
                window.document.body.textContent = msg
            else:
                window.console.info(msg)

        def on_ws_error(self, evt):
            if False:
                for i in range(10):
                    print('nop')
            self._ws = None
            self.status = 0
            window.console.error('Socket error')
        ws.onopen = on_ws_open
        ws.onmessage = on_ws_message
        ws.onclose = on_ws_close
        ws.onerror = on_ws_error

    def _process_commands(self):
        if False:
            return 10
        ' A less direct way to process commands, which gives the\n        browser time to draw about every other JS asset. This is a\n        tradeoff between a smooth spinner and fast load time.\n        '
        while self._pending_commands is not None and len(self._pending_commands) > 0:
            msg = self._pending_commands.pop(0)
            try:
                command = self._receive_raw_command(msg)
            except Exception as err:
                window.setTimeout(self._process_commands, 0)
                raise err
            if command[0] == 'DEFINE':
                self._asset_count += 1
                if self._asset_count % 3 == 0:
                    if len(self._pending_commands):
                        window.setTimeout(self._process_commands, 0)
                    break

    def _receive_raw_command(self, msg):
        if False:
            i = 10
            return i + 15
        return self._receive_command(serializer.decode(msg))

    def _receive_command(self, command):
        if False:
            while True:
                i = 10
        ' Process a command send from the server.\n        '
        cmd = command[0]
        if cmd == 'PING':
            window.setTimeout(self.send_command, 10, 'PONG', command[1])
        elif cmd == 'INIT_DONE':
            window.flexx.spin(None)
            while len(self._pending_commands):
                self._receive_raw_command(self._pending_commands.pop(0))
            self._pending_commands = None
        elif cmd == 'PRINT':
            (window.console.ori_log or window.console.log)(command[1])
        elif cmd == 'EXEC':
            eval(command[1])
        elif cmd == 'EVAL':
            x = None
            if len(command) == 2:
                x = eval(command[1])
            elif len(command) == 3:
                x = eval('this.instances.' + command[1] + '.' + command[2])
            console.log(str(x))
        elif cmd == 'EVALANDRETURN':
            try:
                x = eval(command[1])
            except Exception as err:
                x = str(err)
            eval_id = command[2]
            self.send_command('EVALRESULT', x, eval_id)
        elif cmd == 'INVOKE':
            (id, name, args) = command[1:]
            ob = self.instances.get(id, None)
            if ob is None:
                console.warn('Cannot invoke %s.%s; session does not know it (anymore).' % (id, name))
            elif ob._disposed is True:
                pass
            else:
                ob[name](*args)
        elif cmd == 'INSTANTIATE':
            self.instantiate_component(*command[1:])
        elif cmd == 'DISPOSE':
            id = command[1]
            c = self.instances.get(id, None)
            if c is not None and c._disposed is False:
                c._dispose()
            self.send_command('DISPOSE_ACK', command[1])
            self.instances.pop(id, None)
        elif cmd == 'DISPOSE_ACK':
            self.instances.pop(command[1], None)
        elif cmd == 'DEFINE':
            (kind, name, code) = command[1:]
            window.flexx.spin()
            address = window.location.protocol + '//' + self.ws_url.split('/')[2]
            code += '\n//# sourceURL=%s/flexx/assets/shared/%s\n' % (address, name)
            if kind == 'JS-EVAL':
                eval(code)
            elif kind == 'JS':
                el = window.document.createElement('script')
                el.id = name
                el.innerHTML = code
                window.flexx.asset_node.appendChild(el)
            elif kind == 'CSS':
                el = window.document.createElement('style')
                el.type = 'text/css'
                el.id = name
                el.innerHTML = code
                window.flexx.asset_node.appendChild(el)
            else:
                window.console.error('Dont know how to DEFINE ' + name + ' with "' + kind + '".')
        elif cmd == 'OPEN':
            window.win1 = window.open(command[1], 'new', 'chrome')
        else:
            window.console.error('Invalid command: "' + cmd + '"')
        return command

    def call_after_roundtrip(self, callback, *args):
        if False:
            i = 10
            return i + 15
        ping_to_schedule_at = self._ping_counter + 1
        if len(self._ping_calls) == 0 or self._ping_calls[-1][0] < ping_to_schedule_at:
            window.setTimeout(self._send_ping, 0)
        self._ping_calls.push((ping_to_schedule_at, callback, args))

    def _send_ping(self):
        if False:
            print('Hello World!')
        self._ping_counter += 1
        self.send_command('PING', self._ping_counter)

    def _receive_pong(self, count):
        if False:
            for i in range(10):
                print('nop')
        while len(self._ping_calls) > 0 and self._ping_calls[0][0] <= count:
            (_, callback, args) = self._ping_calls.pop(0)
            window.setTimeout(callback, 0, *args)

    def keep_checking_size_of(self, ob, check=True):
        if False:
            print('Hello World!')
        ' This is a service that the session provides.\n        '
        if check:
            self.instances_to_check_size[ob.id] = ob
        else:
            self.instances_to_check_size.pop(ob.id, None)

    def _check_size_of_objects(self):
        if False:
            return 10
        for ob in self.instances_to_check_size.values():
            if ob._disposed is False:
                ob.check_real_size()
if this_is_js():
    window.flexx = Flexx()
    bsdf = RawJS("flexx.require('bsdf')")
    serializer = bsdf.BsdfSerializer()
    window.flexx.serializer = serializer
else:
    from . import bsdf_lite as bsdf
    serializer = bsdf.BsdfLiteSerializer()
    serializer.__module__ = __name__