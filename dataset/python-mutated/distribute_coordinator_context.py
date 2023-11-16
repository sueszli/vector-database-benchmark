"""The context retrieval method for distribute coordinator."""
import threading
_worker_context = threading.local()

def get_current_worker_context():
    if False:
        i = 10
        return i + 15
    'Returns the current task context.'
    try:
        return _worker_context.current
    except AttributeError:
        return None