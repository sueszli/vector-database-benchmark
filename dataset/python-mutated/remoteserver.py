import inspect
import sys
from xmlrpc.server import SimpleXMLRPCServer

def keyword(name=None, tags=(), types=()):
    if False:
        i = 10
        return i + 15
    if callable(name):
        return keyword()(name)

    def deco(func):
        if False:
            i = 10
            return i + 15
        func.robot_name = name
        func.robot_tags = tags
        func.robot_types = types
        return func
    return deco

class RemoteServer(SimpleXMLRPCServer):

    def __init__(self, library, port=8270, port_file=None):
        if False:
            print('Hello World!')
        SimpleXMLRPCServer.__init__(self, ('127.0.0.1', int(port)))
        self.library = library
        self._shutdown = False
        self._register_functions()
        announce_port(self.socket, port_file)
        self.serve_forever()

    def _register_functions(self):
        if False:
            print('Hello World!')
        self.register_function(self.get_keyword_names)
        self.register_function(self.get_keyword_arguments)
        self.register_function(self.get_keyword_tags)
        self.register_function(self.get_keyword_documentation)
        self.register_function(self.run_keyword)

    def serve_forever(self):
        if False:
            return 10
        while not self._shutdown:
            self.handle_request()

    def get_keyword_names(self):
        if False:
            for i in range(10):
                print('nop')
        return [attr for attr in dir(self.library) if attr[0] != '_']

    def get_keyword_arguments(self, name):
        if False:
            i = 10
            return i + 15
        kw = getattr(self.library, name)
        (args, varargs, kwargs, defaults, kwoargs, kwodefaults, _) = inspect.getfullargspec(kw)
        args = args[1:]
        if defaults:
            (args, names) = (args[:-len(defaults)], args[-len(defaults):])
            args += [f'{n}={d}' for (n, d) in zip(names, defaults)]
        if varargs:
            args.append(f'*{varargs}')
        if kwoargs:
            if not varargs:
                args.append('*')
            args += [self._format_kwo(arg, kwodefaults) for arg in kwoargs]
        if kwargs:
            args.append(f'**{kwargs}')
        return args

    def _format_kwo(self, arg, defaults):
        if False:
            return 10
        if defaults and arg in defaults:
            return f'{arg}={defaults[arg]}'
        return arg

    def get_keyword_tags(self, name):
        if False:
            while True:
                i = 10
        kw = getattr(self.library, name)
        return getattr(kw, 'robot_tags', [])

    def get_keyword_documentation(self, name):
        if False:
            i = 10
            return i + 15
        kw = getattr(self.library, name)
        return inspect.getdoc(kw) or ''

    def run_keyword(self, name, args, kwargs=None):
        if False:
            return 10
        try:
            result = getattr(self.library, name)(*args, **kwargs or {})
        except AssertionError as err:
            return {'status': 'FAIL', 'error': str(err)}
        else:
            return {'status': 'PASS', 'return': result if result is not None else ''}

class DirectResultRemoteServer(RemoteServer):

    def run_keyword(self, name, args, kwargs=None):
        if False:
            return 10
        try:
            return getattr(self.library, name)(*args, **kwargs or {})
        except SystemExit:
            self._shutdown = True
            return {'status': 'PASS'}

def announce_port(socket, port_file=None):
    if False:
        i = 10
        return i + 15
    port = socket.getsockname()[1]
    sys.stdout.write(f'Remote server starting on port {port}.\n')
    sys.stdout.flush()
    if port_file:
        with open(port_file, 'w') as f:
            f.write(str(port))