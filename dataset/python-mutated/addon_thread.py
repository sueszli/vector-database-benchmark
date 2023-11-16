from copy import copy
from .plugin_thread import PluginThread

class AddonThread(PluginThread):
    """
    thread for addons.
    """

    def __init__(self, manager, function, args, kwargs):
        if False:
            print('Hello World!')
        '\n        Constructor.\n        '
        super().__init__(manager)
        self.f = function
        self.args = args
        self.kwargs = kwargs
        self.active = []
        manager.local_threads.append(self)
        self.start()

    def get_active_files(self):
        if False:
            return 10
        return self.active

    def add_active(self, pyfile):
        if False:
            print('Hello World!')
        '\n        Adds a pyfile to active list and thus will be displayed on overview.\n        '
        if pyfile not in self.active:
            self.active.append(pyfile)

    def finish_file(self, pyfile):
        if False:
            i = 10
            return i + 15
        if pyfile in self.active:
            self.active.remove(pyfile)
        pyfile.finish_if_done()

    def run(self):
        if False:
            print('Hello World!')
        try:
            retry = False
            try:
                self.kwargs['thread'] = self
                self.f(*self.args, **self.kwargs)
            except TypeError as exc:
                if "unexpected keyword argument 'thread'" not in exc.args[0]:
                    raise
                else:
                    retry = True
            if retry:
                del self.kwargs['thread']
                self.f(*self.args, **self.kwargs)
        finally:
            local = copy(self.active)
            for x in local:
                self.finish_file(x)
            self.m.local_threads.remove(self)