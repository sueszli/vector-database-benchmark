__all__ = ['Dialog']
from tkinter import Frame, _get_temp_root, _destroy_temp_root

class Dialog:
    command = None

    def __init__(self, master=None, **options):
        if False:
            for i in range(10):
                print('nop')
        if master is None:
            master = options.get('parent')
        self.master = master
        self.options = options

    def _fixoptions(self):
        if False:
            print('Hello World!')
        pass

    def _fixresult(self, widget, result):
        if False:
            return 10
        return result

    def show(self, **options):
        if False:
            print('Hello World!')
        for (k, v) in options.items():
            self.options[k] = v
        self._fixoptions()
        master = self.master
        if master is None:
            master = _get_temp_root()
        try:
            self._test_callback(master)
            s = master.tk.call(self.command, *master._options(self.options))
            s = self._fixresult(master, s)
        finally:
            _destroy_temp_root(master)
        return s

    def _test_callback(self, master):
        if False:
            for i in range(10):
                print('nop')
        pass