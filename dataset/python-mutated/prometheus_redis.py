import os
import socket
from django.views import View
from django.shortcuts import render
'\nRedisActiveConnections\nRedisCommands\nRedisConnects\nRedisUsedMemory\nRedisSize\n'

class RedisGrafanaMetric(View):
    category = 'Redis'

    def autoconf(self):
        if False:
            while True:
                i = 10
        try:
            self.get_info()
        except socket.error:
            return False
        return True

    def get_info(self):
        if False:
            i = 10
            return i + 15
        host = os.environ.get('REDIS_HOST') or '127.0.0.1'
        port = int(os.environ.get('REDIS_PORT') or '6379')
        if host.startswith('/'):
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(host)
        else:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
        s.send('*1\r\n$4\r\ninfo\r\n')
        buf = ''
        while '\r\n' not in buf:
            buf += s.recv(1024)
        (l, buf) = buf.split('\r\n', 1)
        if l[0] != '$':
            s.close()
            raise Exception('Protocol error')
        remaining = int(l[1:]) - len(buf)
        if remaining > 0:
            buf += s.recv(remaining)
        s.close()
        return dict((x.split(':', 1) for x in buf.split('\r\n') if ':' in x))

    def execute(self):
        if False:
            print('Hello World!')
        stats = self.get_info()
        values = {}
        for (k, v) in self.get_fields():
            try:
                value = stats[k]
            except KeyError:
                value = 'U'
            values[k] = value
        return values

    def get_fields(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('You must implement the get_fields function')

    def get_context(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('You must implement the get_context function')

    def get(self, request):
        if False:
            return 10
        context = self.get_context()
        return render(request, 'monitor/prometheus_data.html', context, content_type='text/plain')

class RedisActiveConnection(RedisGrafanaMetric):

    def get_fields(self):
        if False:
            print('Hello World!')
        return (('connected_clients', dict(label='connections', info='connections', type='GAUGE')),)

    def get_context(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('You must implement the get_context function')