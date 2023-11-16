import paramiko
import queue
import urllib.parse
import requests.adapters
import logging
import os
import signal
import socket
import subprocess
from docker.transport.basehttpadapter import BaseHTTPAdapter
from .. import constants
import urllib3
import urllib3.connection
RecentlyUsedContainer = urllib3._collections.RecentlyUsedContainer

class SSHSocket(socket.socket):

    def __init__(self, host):
        if False:
            return 10
        super().__init__(socket.AF_INET, socket.SOCK_STREAM)
        self.host = host
        self.port = None
        self.user = None
        if ':' in self.host:
            (self.host, self.port) = self.host.split(':')
        if '@' in self.host:
            (self.user, self.host) = self.host.split('@')
        self.proc = None

    def connect(self, **kwargs):
        if False:
            while True:
                i = 10
        args = ['ssh']
        if self.user:
            args = args + ['-l', self.user]
        if self.port:
            args = args + ['-p', self.port]
        args = args + ['--', self.host, 'docker system dial-stdio']
        preexec_func = None
        if not constants.IS_WINDOWS_PLATFORM:

            def f():
                if False:
                    return 10
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            preexec_func = f
        env = dict(os.environ)
        env.pop('LD_LIBRARY_PATH', None)
        env.pop('SSL_CERT_FILE', None)
        self.proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stdin=subprocess.PIPE, preexec_fn=preexec_func)

    def _write(self, data):
        if False:
            return 10
        if not self.proc or self.proc.stdin.closed:
            raise Exception('SSH subprocess not initiated.connect() must be called first.')
        written = self.proc.stdin.write(data)
        self.proc.stdin.flush()
        return written

    def sendall(self, data):
        if False:
            print('Hello World!')
        self._write(data)

    def send(self, data):
        if False:
            print('Hello World!')
        return self._write(data)

    def recv(self, n):
        if False:
            print('Hello World!')
        if not self.proc:
            raise Exception('SSH subprocess not initiated.connect() must be called first.')
        return self.proc.stdout.read(n)

    def makefile(self, mode):
        if False:
            for i in range(10):
                print('nop')
        if not self.proc:
            self.connect()
        self.proc.stdout.channel = self
        return self.proc.stdout

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.proc or self.proc.stdin.closed:
            return
        self.proc.stdin.write(b'\n\n')
        self.proc.stdin.flush()
        self.proc.terminate()

class SSHConnection(urllib3.connection.HTTPConnection):

    def __init__(self, ssh_transport=None, timeout=60, host=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('localhost', timeout=timeout)
        self.ssh_transport = ssh_transport
        self.timeout = timeout
        self.ssh_host = host

    def connect(self):
        if False:
            return 10
        if self.ssh_transport:
            sock = self.ssh_transport.open_session()
            sock.settimeout(self.timeout)
            sock.exec_command('docker system dial-stdio')
        else:
            sock = SSHSocket(self.ssh_host)
            sock.settimeout(self.timeout)
            sock.connect()
        self.sock = sock

class SSHConnectionPool(urllib3.connectionpool.HTTPConnectionPool):
    scheme = 'ssh'

    def __init__(self, ssh_client=None, timeout=60, maxsize=10, host=None):
        if False:
            print('Hello World!')
        super().__init__('localhost', timeout=timeout, maxsize=maxsize)
        self.ssh_transport = None
        self.timeout = timeout
        if ssh_client:
            self.ssh_transport = ssh_client.get_transport()
        self.ssh_host = host

    def _new_conn(self):
        if False:
            for i in range(10):
                print('nop')
        return SSHConnection(self.ssh_transport, self.timeout, self.ssh_host)

    def _get_conn(self, timeout):
        if False:
            i = 10
            return i + 15
        conn = None
        try:
            conn = self.pool.get(block=self.block, timeout=timeout)
        except AttributeError as ae:
            raise urllib3.exceptions.ClosedPoolError(self, 'Pool is closed.') from ae
        except queue.Empty:
            if self.block:
                raise urllib3.exceptions.EmptyPoolError(self, 'Pool reached maximum size and no more connections are allowed.') from None
        return conn or self._new_conn()

class SSHHTTPAdapter(BaseHTTPAdapter):
    __attrs__ = requests.adapters.HTTPAdapter.__attrs__ + ['pools', 'timeout', 'ssh_client', 'ssh_params', 'max_pool_size']

    def __init__(self, base_url, timeout=60, pool_connections=constants.DEFAULT_NUM_POOLS, max_pool_size=constants.DEFAULT_MAX_POOL_SIZE, shell_out=False):
        if False:
            while True:
                i = 10
        self.ssh_client = None
        if not shell_out:
            self._create_paramiko_client(base_url)
            self._connect()
        self.ssh_host = base_url
        if base_url.startswith('ssh://'):
            self.ssh_host = base_url[len('ssh://'):]
        self.timeout = timeout
        self.max_pool_size = max_pool_size
        self.pools = RecentlyUsedContainer(pool_connections, dispose_func=lambda p: p.close())
        super().__init__()

    def _create_paramiko_client(self, base_url):
        if False:
            return 10
        logging.getLogger('paramiko').setLevel(logging.WARNING)
        self.ssh_client = paramiko.SSHClient()
        base_url = urllib.parse.urlparse(base_url)
        self.ssh_params = {'hostname': base_url.hostname, 'port': base_url.port, 'username': base_url.username}
        ssh_config_file = os.path.expanduser('~/.ssh/config')
        if os.path.exists(ssh_config_file):
            conf = paramiko.SSHConfig()
            with open(ssh_config_file) as f:
                conf.parse(f)
            host_config = conf.lookup(base_url.hostname)
            if 'proxycommand' in host_config:
                self.ssh_params['sock'] = paramiko.ProxyCommand(host_config['proxycommand'])
            if 'hostname' in host_config:
                self.ssh_params['hostname'] = host_config['hostname']
            if base_url.port is None and 'port' in host_config:
                self.ssh_params['port'] = host_config['port']
            if base_url.username is None and 'user' in host_config:
                self.ssh_params['username'] = host_config['user']
            if 'identityfile' in host_config:
                self.ssh_params['key_filename'] = host_config['identityfile']
        self.ssh_client.load_system_host_keys()
        self.ssh_client.set_missing_host_key_policy(paramiko.RejectPolicy())

    def _connect(self):
        if False:
            for i in range(10):
                print('nop')
        if self.ssh_client:
            self.ssh_client.connect(**self.ssh_params)

    def get_connection(self, url, proxies=None):
        if False:
            print('Hello World!')
        if not self.ssh_client:
            return SSHConnectionPool(ssh_client=self.ssh_client, timeout=self.timeout, maxsize=self.max_pool_size, host=self.ssh_host)
        with self.pools.lock:
            pool = self.pools.get(url)
            if pool:
                return pool
            if self.ssh_client and (not self.ssh_client.get_transport()):
                self._connect()
            pool = SSHConnectionPool(ssh_client=self.ssh_client, timeout=self.timeout, maxsize=self.max_pool_size, host=self.ssh_host)
            self.pools[url] = pool
        return pool

    def close(self):
        if False:
            while True:
                i = 10
        super().close()
        if self.ssh_client:
            self.ssh_client.close()