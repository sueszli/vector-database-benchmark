import time
import urllib.request as request
from threading import Thread
from dramatiq.middleware.prometheus import _run_exposition_server

def test_prometheus_middleware_exposes_metrics():
    if False:
        for i in range(10):
            print('nop')
    thread = Thread(target=_run_exposition_server, daemon=True)
    thread.start()
    time.sleep(1)
    with request.urlopen('http://127.0.0.1:9191') as resp:
        assert resp.getcode() == 200