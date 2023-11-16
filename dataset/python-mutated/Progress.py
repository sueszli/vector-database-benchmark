""" Progress bars in Nuitka.

This is responsible for wrapping the rendering of progress bar and emitting tracing
to the user while it's being displayed.

"""
from contextlib import contextmanager
from nuitka import Tracing
from nuitka.Tracing import general
from nuitka.utils.Importing import importFromInlineCopy
from nuitka.utils.ThreadedExecutor import RLock
from nuitka.utils.Utils import isWin32Windows
use_progress_bar = False
tqdm = None
colorama = None

class NuitkaProgressBar(object):

    def __init__(self, iterable, stage, total, min_total, unit):
        if False:
            while True:
                i = 10
        self.total = total
        self.min_total = min_total
        self.item = None
        self.progress = 0
        self.tqdm = tqdm(iterable=iterable, initial=self.progress, total=max(self.total, self.min_total) if self.min_total is not None else None, unit=unit, disable=None, leave=False, dynamic_ncols=True, bar_format='{desc}{percentage:3.1f}%|{bar:25}| {n_fmt}/{total_fmt}{postfix}')
        self.tqdm.set_description(stage)
        self.setCurrent(self.item)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.tqdm)

    def updateTotal(self, total):
        if False:
            while True:
                i = 10
        if total != self.total:
            self.total = total
            self.tqdm.total = max(total, self.min_total)

    def setCurrent(self, item):
        if False:
            for i in range(10):
                print('nop')
        if item != self.item:
            self.item = item
            if item is not None:
                self.tqdm.set_postfix_str(item)
            else:
                self.tqdm.set_postfix()

    def update(self):
        if False:
            print('Hello World!')
        self.progress += 1
        self.tqdm.update(1)

    def clear(self):
        if False:
            i = 10
            return i + 15
        self.tqdm.clear()

    def close(self):
        if False:
            while True:
                i = 10
        self.tqdm.close()

    @contextmanager
    def withExternalWritingPause(self):
        if False:
            return 10
        with self.tqdm.external_write_mode():
            yield

def _getTqdmModule():
    if False:
        print('Hello World!')
    global tqdm
    if tqdm:
        return tqdm
    elif tqdm is False:
        return None
    else:
        tqdm = importFromInlineCopy('tqdm', must_exist=False, delete_module=True)
        if tqdm is None:
            try:
                import tqdm as tqdm_installed
                tqdm = tqdm_installed
            except ImportError:
                pass
        if tqdm is None:
            tqdm = False
            return None
        tqdm = tqdm.tqdm
        tqdm.set_lock(RLock())
        return tqdm

def enableProgressBar():
    if False:
        while True:
            i = 10
    global use_progress_bar
    global colorama
    if _getTqdmModule() is not None:
        use_progress_bar = True
        if isWin32Windows():
            if colorama is None:
                colorama = importFromInlineCopy('colorama', must_exist=True, delete_module=True)
            colorama.init()

def setupProgressBar(stage, unit, total, min_total=0):
    if False:
        while True:
            i = 10
    assert Tracing.progress is None
    if use_progress_bar:
        Tracing.progress = NuitkaProgressBar(iterable=None, stage=stage, total=total, min_total=min_total, unit=unit)

def reportProgressBar(item, total=None, update=True):
    if False:
        print('Hello World!')
    if Tracing.progress is not None:
        try:
            if total is not None:
                Tracing.progress.updateTotal(total)
            Tracing.progress.setCurrent(item)
            if update:
                Tracing.progress.update()
        except Exception as e:
            general.warning('Progress bar disabled due to bug: %s' % str(e))
            closeProgressBar()

def closeProgressBar():
    if False:
        print('Hello World!')
    'Close the active progress bar.\n\n    Returns: int or None - if displayed, the total used last time.\n    '
    if Tracing.progress is not None:
        result = Tracing.progress.total
        Tracing.progress.close()
        Tracing.progress = None
        return result

def wrapWithProgressBar(iterable, stage, unit):
    if False:
        while True:
            i = 10
    if tqdm is None:
        return iterable
    else:
        result = NuitkaProgressBar(iterable=iterable, unit=unit, stage=stage, total=None, min_total=None)
        Tracing.progress = result
        return result

@contextmanager
def withNuitkaDownloadProgressBar(*args, **kwargs):
    if False:
        return 10
    if not use_progress_bar or _getTqdmModule() is None:
        yield
    else:

        class NuitkaDownloadProgressBar(tqdm):

            def onProgress(self, b=1, bsize=1, tsize=None):
                if False:
                    while True:
                        i = 10
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        kwargs.update(disable=None, leave=False, dynamic_ncols=True, bar_format='{desc} {percentage:3.1f}%|{bar:25}| {n_fmt}/{total_fmt}{postfix}')
        with NuitkaDownloadProgressBar(*args, **kwargs) as progress_bar:
            yield progress_bar.onProgress