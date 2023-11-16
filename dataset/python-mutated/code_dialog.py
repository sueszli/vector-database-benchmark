import functools
import threading
import tkinter as tk
from tkinter import StringVar, messagebox
from tkinter import ttk

class CodeDialog(object):

    def __init__(self, title=None, label='Code Dialog'):
        if False:
            for i in range(10):
                print('nop')
        self.root = tk.Tk()
        self.root.title(title)
        self.code = StringVar()
        mainframe = ttk.Frame(self.root, padding='12 12 12 12')
        mainframe.grid(column=0, row=0)
        self.label = ttk.Label(mainframe, text=label, width=10)
        self.input = ttk.Entry(mainframe, textvariable=self.code, width=20)
        self.button = ttk.Button(mainframe, text='ok', command=self.click_ok, width=5)
        self.label.grid(row=1, column=0)
        self.input.grid(row=1, column=1)
        self.button.grid(row=2, column=1, sticky=tk.E)
        self.root.bind('<Return>', self.click_ok)

    def wait_string(self):
        if False:
            while True:
                i = 10
        self.root.eval('tk::PlaceWindow . center')
        self.root.mainloop()
        return self.code.get()

    def click_ok(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if not self.code.get():
            messagebox.showwarning(title='warning', message='code is empty')
            return
        self.root.destroy()

class TkProgressBar(object):
    wait_max_time = 3000 * 10

    def __init__(self, wait_func=None):
        if False:
            return 10
        self._wait_func = wait_func
        self._done = threading.Event()
        self._root = None
        self._wait_time = 0

    def _check(self):
        if False:
            for i in range(10):
                print('nop')
        if self._done.isSet():
            self._root.destroy()
            return
        if self._wait_time >= self.wait_max_time:
            self._root.destroy()
            return
        self._wait_time += 100
        self._root.after(100, self._check)

    def stop(self):
        if False:
            return 10
        self._done.set()

    def show(self):
        if False:
            print('Hello World!')
        if not self._wait_func:
            return
        root = tk.Tk()
        (width, height) = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.geometry('%dx%d' % (width, height))
        root.title('Progress')
        root.grid()
        pb_length = width - 20
        pb = ttk.Progressbar(root, orient='horizontal', length=pb_length, mode='indeterminate')
        pb.pack(expand=True, padx=10, pady=10)
        self._root = root
        self._root.after(60, pb.start)
        self._root.after(100, self._check)

        def _wait_run_func():
            if False:
                return 10
            self._wait_func()
            self.stop()
            print('wait func done')
        self._root.attributes('-topmost', True)
        self._root.attributes('-fullscreen', True)
        threading.Thread(target=_wait_run_func).start()
        self._root.mainloop()

def wrapper_progress_bar(func):
    if False:
        print('Hello World!')

    def inner(*args, **kwargs):
        if False:
            return 10
        wait_func = functools.partial(func, *args, **kwargs)
        tk_process = TkProgressBar(wait_func=wait_func)
        tk_process.show()
    return inner
if __name__ == '__main__':
    import time
    progress = TkProgressBar(wait_func=lambda : time.sleep(15))
    progress.show()
    print('end')