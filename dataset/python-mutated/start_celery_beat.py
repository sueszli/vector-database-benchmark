import os
import sys
import signal
import subprocess
import redis_lock
from redis import Redis, Sentinel
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPS_DIR = os.path.join(BASE_DIR, 'apps')
CERTS_DIR = os.path.join(BASE_DIR, 'data', 'certs')
sys.path.insert(0, APPS_DIR)
from jumpserver import settings
os.environ.setdefault('PYTHONOPTIMIZE', '1')
if os.getuid() == 0:
    os.environ.setdefault('C_FORCE_ROOT', '1')
connection_params = {'password': settings.REDIS_PASSWORD}
if settings.REDIS_USE_SSL:
    connection_params['ssl'] = settings.REDIS_USE_SSL
    connection_params['ssl_cert_reqs'] = settings.REDIS_SSL_REQUIRED
    connection_params['ssl_keyfile'] = settings.REDIS_SSL_KEY
    connection_params['ssl_certfile'] = settings.REDIS_SSL_CERT
    connection_params['ssl_ca_certs'] = settings.REDIS_SSL_CA
REDIS_SENTINEL_SERVICE_NAME = settings.REDIS_SENTINEL_SERVICE_NAME
REDIS_SENTINELS = settings.REDIS_SENTINELS
REDIS_SENTINEL_PASSWORD = settings.REDIS_SENTINEL_PASSWORD
REDIS_SENTINEL_SOCKET_TIMEOUT = settings.REDIS_SENTINEL_SOCKET_TIMEOUT
if REDIS_SENTINEL_SERVICE_NAME and REDIS_SENTINELS:
    connection_params['sentinels'] = REDIS_SENTINELS
    sentinel_client = Sentinel(**connection_params, sentinel_kwargs={'ssl': settings.REDIS_USE_SSL, 'ssl_cert_reqs': settings.REDIS_SSL_REQUIRED, 'ssl_keyfile': settings.REDIS_SSL_KEY, 'ssl_certfile': settings.REDIS_SSL_CERT, 'ssl_ca_certs': settings.REDIS_SSL_CA, 'password': REDIS_SENTINEL_PASSWORD, 'socket_timeout': REDIS_SENTINEL_SOCKET_TIMEOUT})
    redis_client = sentinel_client.master_for(REDIS_SENTINEL_SERVICE_NAME)
else:
    connection_params['host'] = settings.REDIS_HOST
    connection_params['port'] = settings.REDIS_PORT
    redis_client = Redis(**connection_params)
scheduler = 'ops.celery.beat.schedulers:DatabaseScheduler'
processes = []
cmd = ['celery', '-A', 'ops', 'beat', '-l', 'INFO', '--scheduler', scheduler, '--max-interval', '60']

def stop_beat_process(sig, frame):
    if False:
        print('Hello World!')
    for p in processes:
        os.kill(p.pid, 15)

def main():
    if False:
        return 10
    signal.signal(signal.SIGTERM, stop_beat_process)
    with redis_lock.Lock(redis_client, name='beat-distribute-start-lock', expire=60, auto_renewal=True):
        print('Get beat lock start to run it')
        process = subprocess.Popen(cmd, cwd=APPS_DIR)
        processes.append(process)
        process.wait()
if __name__ == '__main__':
    main()