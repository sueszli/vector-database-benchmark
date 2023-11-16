import jesse.helpers as jh
import jesse.services.logger as logger
from jesse.store import store

def set_up():
    if False:
        i = 10
        return i + 15
    store.reset()