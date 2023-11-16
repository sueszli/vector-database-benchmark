from threading import RLock
import cloudpickle
_lock = RLock()

def loads(payload):
    if False:
        return 10
    with _lock:
        return cloudpickle.loads(payload)