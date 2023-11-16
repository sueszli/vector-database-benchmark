bind = '127.0.0.1:8000'
backlog = 2048
workers = 1
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2
spew = False
daemon = False
raw_env = ['DJANGO_SECRET_KEY=something', 'SPAM=eggs']
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None
errorlog = '-'
loglevel = 'info'
accesslog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
proc_name = None

def post_fork(server, worker):
    if False:
        i = 10
        return i + 15
    server.log.info('Worker spawned (pid: %s)', worker.pid)

def pre_fork(server, worker):
    if False:
        for i in range(10):
            print('nop')
    pass

def pre_exec(server):
    if False:
        i = 10
        return i + 15
    server.log.info('Forked child, re-executing.')

def when_ready(server):
    if False:
        return 10
    server.log.info('Server is ready. Spawning workers')

def worker_int(worker):
    if False:
        print('Hello World!')
    worker.log.info('worker received INT or QUIT signal')
    import threading, sys, traceback
    id2name = {th.ident: th.name for th in threading.enumerate()}
    code = []
    for (threadId, stack) in sys._current_frames().items():
        code.append('\n# Thread: %s(%d)' % (id2name.get(threadId, ''), threadId))
        for (filename, lineno, name, line) in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append('  %s' % line.strip())
    worker.log.debug('\n'.join(code))

def worker_abort(worker):
    if False:
        return 10
    worker.log.info('worker received SIGABRT signal')

def ssl_context(conf, default_ssl_context_factory):
    if False:
        for i in range(10):
            print('nop')
    import ssl
    context = default_ssl_context_factory()
    context.minimum_version = ssl.TLSVersion.TLSv1_3

    def sni_callback(socket, server_hostname, context):
        if False:
            for i in range(10):
                print('nop')
        if server_hostname == 'foo.127.0.0.1.nip.io':
            new_context = default_ssl_context_factory()
            new_context.load_cert_chain(certfile='foo.pem', keyfile='foo-key.pem')
            socket.context = new_context
    context.sni_callback = sni_callback
    return context