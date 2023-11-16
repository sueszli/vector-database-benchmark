TIMEOUT = 10
import threading
import tempfile
import os.path
errors = []

class Worker(threading.Thread):

    def __init__(self, function, args):
        if False:
            i = 10
            return i + 15
        threading.Thread.__init__(self)
        self.function = function
        self.args = args

    def run(self):
        if False:
            while True:
                i = 10
        self.function(*self.args)
for (name, func, args) in [('tempfile.TemporaryFile', lambda : tempfile.TemporaryFile().close(), ()), ('os.path.abspath', os.path.abspath, ('.',))]:
    try:
        t = Worker(func, args)
        t.start()
        t.join(TIMEOUT)
        if t.is_alive():
            errors.append('%s appeared to hang' % name)
    finally:
        del t