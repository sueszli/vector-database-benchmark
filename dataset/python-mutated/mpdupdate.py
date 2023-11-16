"""Updates an MPD index whenever the library is changed.

Put something like the following in your config.yaml to configure:
    mpd:
        host: localhost
        port: 6600
        password: seekrit
"""
import os
import socket
from beets import config
from beets.plugins import BeetsPlugin

class BufferedSocket:
    """Socket abstraction that allows reading by line."""

    def __init__(self, host, port, sep=b'\n'):
        if False:
            for i in range(10):
                print('nop')
        if host[0] in ['/', '~']:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(os.path.expanduser(host))
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((host, port))
        self.buf = b''
        self.sep = sep

    def readline(self):
        if False:
            print('Hello World!')
        while self.sep not in self.buf:
            data = self.sock.recv(1024)
            if not data:
                break
            self.buf += data
        if self.sep in self.buf:
            (res, self.buf) = self.buf.split(self.sep, 1)
            return res + self.sep
        else:
            return b''

    def send(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.sock.send(data)

    def close(self):
        if False:
            return 10
        self.sock.close()

class MPDUpdatePlugin(BeetsPlugin):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        config['mpd'].add({'host': os.environ.get('MPD_HOST', 'localhost'), 'port': int(os.environ.get('MPD_PORT', 6600)), 'password': ''})
        config['mpd']['password'].redact = True
        for key in config['mpd'].keys():
            if self.config[key].exists():
                config['mpd'][key] = self.config[key].get()
        self.register_listener('database_change', self.db_change)

    def db_change(self, lib, model):
        if False:
            while True:
                i = 10
        self.register_listener('cli_exit', self.update)

    def update(self, lib):
        if False:
            i = 10
            return i + 15
        self.update_mpd(config['mpd']['host'].as_str(), config['mpd']['port'].get(int), config['mpd']['password'].as_str())

    def update_mpd(self, host='localhost', port=6600, password=None):
        if False:
            for i in range(10):
                print('nop')
        'Sends the "update" command to the MPD server indicated,\n        possibly authenticating with a password first.\n        '
        self._log.info('Updating MPD database...')
        try:
            s = BufferedSocket(host, port)
        except OSError as e:
            self._log.warning('MPD connection failed: {0}', str(e.strerror))
            return
        resp = s.readline()
        if b'OK MPD' not in resp:
            self._log.warning('MPD connection failed: {0!r}', resp)
            return
        if password:
            s.send(b'password "%s"\n' % password.encode('utf8'))
            resp = s.readline()
            if b'OK' not in resp:
                self._log.warning('Authentication failed: {0!r}', resp)
                s.send(b'close\n')
                s.close()
                return
        s.send(b'update\n')
        resp = s.readline()
        if b'updating_db' not in resp:
            self._log.warning('Update failed: {0!r}', resp)
        s.send(b'close\n')
        s.close()
        self._log.info('Database updated.')