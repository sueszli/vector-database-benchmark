import datetime
import errno
import logging
import re
import socket
import time
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream
import config
import motionctl
import settings
import utils

class MjpgClient(IOStream):
    _FPS_LEN = 10
    clients = {}
    _last_erroneous_close_time = 0

    def __init__(self, camera_id, port, username, password, auth_mode):
        if False:
            for i in range(10):
                print('nop')
        self._camera_id = camera_id
        self._port = port
        self._username = (username or '').encode('utf8')
        self._password = (password or '').encode('utf8')
        self._auth_mode = auth_mode
        self._auth_digest_state = {}
        self._last_access = 0
        self._last_jpg = None
        self._last_jpg_times = []
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        IOStream.__init__(self, s)
        self.set_close_callback(self.on_close)

    def do_connect(self):
        if False:
            return 10
        IOStream.connect(self, ('localhost', self._port), self._on_connect)

    def get_port(self):
        if False:
            i = 10
            return i + 15
        return self._port

    def on_close(self):
        if False:
            print('Hello World!')
        logging.debug('connection closed for mjpg client for camera %(camera_id)s on port %(port)s' % {'port': self._port, 'camera_id': self._camera_id})
        if MjpgClient.clients.pop(self._camera_id, None):
            logging.debug('mjpg client for camera %(camera_id)s on port %(port)s removed' % {'port': self._port, 'camera_id': self._camera_id})
        if getattr(self, 'error', None) and self.error.errno != errno.ECONNREFUSED:
            now = time.time()
            if now - MjpgClient._last_erroneous_close_time < settings.MJPG_CLIENT_TIMEOUT:
                msg = 'connection problem detected for mjpg client for camera %(camera_id)s on port %(port)s' % {'port': self._port, 'camera_id': self._camera_id}
                logging.error(msg)
                if settings.MOTION_RESTART_ON_ERRORS:
                    motionctl.stop(invalidate=True)
                    motionctl.start(deferred=True)
            MjpgClient._last_erroneous_close_time = now

    def get_last_jpg(self):
        if False:
            return 10
        self._last_access = time.time()
        return self._last_jpg

    def get_last_access(self):
        if False:
            while True:
                i = 10
        return self._last_access

    def get_last_jpg_time(self):
        if False:
            while True:
                i = 10
        if not self._last_jpg_times:
            self._last_jpg_times.append(time.time())
        return self._last_jpg_times[-1]

    def get_fps(self):
        if False:
            return 10
        if len(self._last_jpg_times) < self._FPS_LEN:
            return 0
        if time.time() - self._last_jpg_times[-1] > 1:
            return 0
        return (len(self._last_jpg_times) - 1) / (self._last_jpg_times[-1] - self._last_jpg_times[0])

    def _check_error(self):
        if False:
            while True:
                i = 10
        if self.socket is None:
            logging.warning('mjpg client connection for camera %(camera_id)s on port %(port)s is closed' % {'port': self._port, 'camera_id': self._camera_id})
            self.close()
            return True
        error = getattr(self, 'error', None)
        if error is None or getattr(error, 'errno', None) == 0:
            return False
        self._error(error)
        return True

    def _error(self, error):
        if False:
            for i in range(10):
                print('nop')
        logging.error('mjpg client for camera %(camera_id)s on port %(port)s error: %(msg)s' % {'port': self._port, 'camera_id': self._camera_id, 'msg': unicode(error)})
        try:
            self.close()
        except:
            pass

    def _on_connect(self):
        if False:
            while True:
                i = 10
        logging.debug('mjpg client for camera %(camera_id)s connected on port %(port)s' % {'port': self._port, 'camera_id': self._camera_id})
        if self._auth_mode == 'basic':
            logging.debug('mjpg client using basic authentication')
            auth_header = utils.build_basic_header(self._username, self._password)
            self.write('GET / HTTP/1.1\r\nAuthorization: %s\r\nConnection: close\r\n\r\n' % auth_header)
        elif self._auth_mode == 'digest':
            self.write('GET / HTTP/1.1\r\n\r\n')
        else:
            self.write('GET / HTTP/1.1\r\nConnection: close\r\n\r\n')
        self._seek_http()

    def _seek_http(self):
        if False:
            print('Hello World!')
        if self._check_error():
            return
        self.read_until_regex('HTTP/1.\\d \\d+ ', self._on_http)

    def _on_http(self, data):
        if False:
            i = 10
            return i + 15
        if data.endswith('401 '):
            self._seek_www_authenticate()
        else:
            self._seek_content_length()

    def _seek_www_authenticate(self):
        if False:
            i = 10
            return i + 15
        if self._check_error():
            return
        self.read_until('WWW-Authenticate:', self._on_before_www_authenticate)

    def _on_before_www_authenticate(self, data):
        if False:
            for i in range(10):
                print('nop')
        if self._check_error():
            return
        self.read_until('\r\n', self._on_www_authenticate)

    def _on_www_authenticate(self, data):
        if False:
            while True:
                i = 10
        if self._check_error():
            return
        data = data.strip()
        m = re.match('Basic\\s*realm="([a-zA-Z0-9\\-\\s]+)"', data)
        if m:
            logging.debug('mjpg client using basic authentication')
            auth_header = utils.build_basic_header(self._username, self._password)
            self.write('GET / HTTP/1.1\r\nAuthorization: %s\r\nConnection: close\r\n\r\n' % auth_header)
            self._seek_http()
            return
        if data.startswith('Digest'):
            logging.debug('mjpg client using digest authentication')
            parts = data[7:].split(',')
            parts_dict = dict((p.split('=', 1) for p in parts))
            parts_dict = {p[0]: p[1].strip('"') for p in parts_dict.items()}
            self._auth_digest_state = parts_dict
            auth_header = utils.build_digest_header('GET', '/', self._username, self._password, self._auth_digest_state)
            self.write('GET / HTTP/1.1\r\nAuthorization: %s\r\nConnection: close\r\n\r\n' % auth_header)
            self._seek_http()
            return
        logging.error('mjpg client unknown authentication header: "%s"' % data)
        self._seek_content_length()

    def _seek_content_length(self):
        if False:
            return 10
        if self._check_error():
            return
        self.read_until('Content-Length:', self._on_before_content_length)

    def _on_before_content_length(self, data):
        if False:
            for i in range(10):
                print('nop')
        if self._check_error():
            return
        self.read_until('\r\n\r\n', self._on_content_length)

    def _on_content_length(self, data):
        if False:
            for i in range(10):
                print('nop')
        if self._check_error():
            return
        matches = re.findall('(\\d+)', data)
        if not matches:
            self._error('could not find content length in mjpg header line "%(header)s"' % {'header': data})
            return
        length = int(matches[0])
        self.read_bytes(length, self._on_jpg)

    def _on_jpg(self, data):
        if False:
            return 10
        self._last_jpg = data
        self._last_jpg_times.append(time.time())
        while len(self._last_jpg_times) > self._FPS_LEN:
            self._last_jpg_times.pop(0)
        self._seek_content_length()

def start():
    if False:
        for i in range(10):
            print('nop')
    io_loop = IOLoop.instance()
    io_loop.add_timeout(datetime.timedelta(seconds=settings.MJPG_CLIENT_TIMEOUT), _garbage_collector)

def get_jpg(camera_id):
    if False:
        i = 10
        return i + 15
    if camera_id not in MjpgClient.clients:
        logging.debug('creating mjpg client for camera %(camera_id)s' % {'camera_id': camera_id})
        camera_config = config.get_camera(camera_id)
        if not camera_config['@enabled'] or not utils.is_local_motion_camera(camera_config):
            logging.error('could not start mjpg client for camera id %(camera_id)s: not enabled or not local' % {'camera_id': camera_id})
            return None
        port = camera_config['stream_port']
        (username, password) = (None, None)
        auth_mode = None
        if camera_config.get('stream_auth_method') > 0:
            (username, password) = camera_config.get('stream_authentication', ':').split(':')
            auth_mode = 'digest' if camera_config.get('stream_auth_method') > 1 else 'basic'
        client = MjpgClient(camera_id, port, username, password, auth_mode)
        client.do_connect()
        MjpgClient.clients[camera_id] = client
    client = MjpgClient.clients[camera_id]
    return client.get_last_jpg()

def get_fps(camera_id):
    if False:
        while True:
            i = 10
    client = MjpgClient.clients.get(camera_id)
    if client is None:
        return 0
    return client.get_fps()

def close_all(invalidate=False):
    if False:
        while True:
            i = 10
    for client in MjpgClient.clients.values():
        client.close()
    if invalidate:
        MjpgClient.clients = {}
        MjpgClient._last_erroneous_close_time = 0

def _garbage_collector():
    if False:
        for i in range(10):
            print('nop')
    io_loop = IOLoop.instance()
    io_loop.add_timeout(datetime.timedelta(seconds=settings.MJPG_CLIENT_TIMEOUT), _garbage_collector)
    now = time.time()
    for (camera_id, client) in MjpgClient.clients.items():
        port = client.get_port()
        if client.closed():
            continue
        last_jpg_time = client.get_last_jpg_time()
        delta = now - last_jpg_time
        if delta > settings.MJPG_CLIENT_TIMEOUT:
            logging.error('mjpg client timed out receiving data for camera %(camera_id)s on port %(port)s' % {'camera_id': camera_id, 'port': port})
            if settings.MOTION_RESTART_ON_ERRORS:
                motionctl.stop(invalidate=True)
                motionctl.start(deferred=True)
            break
        delta = now - client.get_last_access()
        if settings.MJPG_CLIENT_IDLE_TIMEOUT and delta > settings.MJPG_CLIENT_IDLE_TIMEOUT:
            msg = 'mjpg client for camera %(camera_id)s on port %(port)s has been idle for %(timeout)s seconds, removing it' % {'camera_id': camera_id, 'port': port, 'timeout': settings.MJPG_CLIENT_IDLE_TIMEOUT}
            logging.debug(msg)
            client.close()
            continue