import gc
import threading
from OpenSSL import SSL
from pympler import muppy
from pympler import refbrowser
step = 0
__memory_locals__ = True

def str_fun(obj):
    if False:
        print('Hello World!')
    if isinstance(obj, dict):
        if '__memory_locals__' in obj:
            return '(-locals-)'
        if 'self' in obj and isinstance(obj['self'], refbrowser.InteractiveBrowser):
            return '(-browser-)'
    return str(id(obj)) + ': ' + str(obj)[:100].replace('\r\n', '\\r\\n').replace('\n', '\\n')

def request(ctx, flow):
    if False:
        i = 10
        return i + 15
    global step, ssl
    print('==========')
    print(f'GC: {gc.collect()}')
    print(f'Threads: {threading.active_count()}')
    step += 1
    if step == 1:
        all_objects = muppy.get_objects()
        ssl = muppy.filter(all_objects, SSL.Connection)[0]
    if step == 2:
        ib = refbrowser.InteractiveBrowser(ssl, 2, str_fun, repeat=False)
        del ssl
        ib.main(True)