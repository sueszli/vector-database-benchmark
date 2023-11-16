__license__ = 'GPL 3'
__copyright__ = '2011, John Schember <john@nachtimwald.com>'
__docformat__ = 'restructuredtext en'
import traceback
from contextlib import closing
from threading import Thread
from calibre import browser
from calibre.constants import DEBUG
from calibre.utils.img import scale_image
from polyglot.queue import Queue
from polyglot.binary import from_base64_bytes

class GenericDownloadThreadPool:
    """
    add_task must be implemented in a subclass and must
    GenericDownloadThreadPool.add_task must be called
    at the end of the function.
    """

    def __init__(self, thread_type, thread_count=1):
        if False:
            while True:
                i = 10
        self.thread_type = thread_type
        self.thread_count = thread_count
        self.tasks = Queue()
        self.results = Queue()
        self.threads = []

    def set_thread_count(self, thread_count):
        if False:
            i = 10
            return i + 15
        self.thread_count = thread_count

    def add_task(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This must be implemented in a sub class and this function\n        must be called at the end of the add_task function in\n        the sub class.\n\n        The implementation of this function (in this base class)\n        starts any threads necessary to fill the pool if it is\n        not already full.\n        '
        for i in range(self.thread_count - self.running_threads_count()):
            t = self.thread_type(self.tasks, self.results)
            self.threads.append(t)
            t.start()

    def abort(self):
        if False:
            while True:
                i = 10
        self.tasks = Queue()
        self.results = Queue()
        for t in self.threads:
            t.abort()
        self.threads = []

    def has_tasks(self):
        if False:
            print('Hello World!')
        return not self.tasks.empty()

    def get_result(self):
        if False:
            print('Hello World!')
        return self.results.get()

    def get_result_no_wait(self):
        if False:
            print('Hello World!')
        return self.results.get_nowait()

    def result_count(self):
        if False:
            return 10
        return len(self.results)

    def has_results(self):
        if False:
            i = 10
            return i + 15
        return not self.results.empty()

    def threads_running(self):
        if False:
            return 10
        return self.running_threads_count() > 0

    def running_threads_count(self):
        if False:
            print('Hello World!')
        count = 0
        for t in self.threads:
            if t.is_alive():
                count += 1
        return count

class SearchThreadPool(GenericDownloadThreadPool):
    """
    Threads will run until there is no work or
    abort is called. Create and start new threads
    using start_threads(). Reset by calling abort().

    Example:
    sp = SearchThreadPool(3)
    sp.add_task(...)
    """

    def __init__(self, thread_count):
        if False:
            i = 10
            return i + 15
        GenericDownloadThreadPool.__init__(self, SearchThread, thread_count)

    def add_task(self, query, store_name, store_plugin, max_results, timeout):
        if False:
            while True:
                i = 10
        self.tasks.put((query, store_name, store_plugin, max_results, timeout))
        GenericDownloadThreadPool.add_task(self)

class SearchThread(Thread):

    def __init__(self, tasks, results):
        if False:
            return 10
        Thread.__init__(self)
        self.daemon = True
        self.tasks = tasks
        self.results = results
        self._run = True

    def abort(self):
        if False:
            for i in range(10):
                print('nop')
        self._run = False

    def run(self):
        if False:
            return 10
        while self._run and (not self.tasks.empty()):
            try:
                (query, store_name, store_plugin, max_results, timeout) = self.tasks.get()
                for res in store_plugin.search(query, max_results=max_results, timeout=timeout):
                    if not self._run:
                        return
                    res.store_name = store_name
                    res.affiliate = store_plugin.base_plugin.affiliate
                    res.plugin_author = store_plugin.base_plugin.author
                    res.create_browser = store_plugin.create_browser
                    self.results.put((res, store_plugin))
                self.tasks.task_done()
            except:
                if DEBUG:
                    traceback.print_exc()

class CoverThreadPool(GenericDownloadThreadPool):

    def __init__(self, thread_count):
        if False:
            i = 10
            return i + 15
        GenericDownloadThreadPool.__init__(self, CoverThread, thread_count)

    def add_task(self, search_result, update_callback, timeout=5):
        if False:
            for i in range(10):
                print('nop')
        self.tasks.put((search_result, update_callback, timeout))
        GenericDownloadThreadPool.add_task(self)

def decode_data_url(url):
    if False:
        return 10
    return from_base64_bytes(url.partition(',')[2])

class CoverThread(Thread):

    def __init__(self, tasks, results):
        if False:
            for i in range(10):
                print('nop')
        Thread.__init__(self)
        self.daemon = True
        self.tasks = tasks
        self.results = results
        self._run = True
        self.br = browser()

    def abort(self):
        if False:
            i = 10
            return i + 15
        self._run = False

    def run(self):
        if False:
            return 10
        while self._run and (not self.tasks.empty()):
            try:
                (result, callback, timeout) = self.tasks.get()
                if result and result.cover_url:
                    if result.cover_url.startswith('data:'):
                        result.cover_data = decode_data_url(result.cover_url)
                    else:
                        with closing(self.br.open(result.cover_url, timeout=timeout)) as f:
                            result.cover_data = f.read()
                    result.cover_data = scale_image(result.cover_data, 256, 256)[2]
                    callback()
                self.tasks.task_done()
            except:
                if DEBUG:
                    traceback.print_exc()

class DetailsThreadPool(GenericDownloadThreadPool):

    def __init__(self, thread_count):
        if False:
            return 10
        GenericDownloadThreadPool.__init__(self, DetailsThread, thread_count)

    def add_task(self, search_result, store_plugin, update_callback, timeout=10):
        if False:
            while True:
                i = 10
        self.tasks.put((search_result, store_plugin, update_callback, timeout))
        GenericDownloadThreadPool.add_task(self)

class DetailsThread(Thread):

    def __init__(self, tasks, results):
        if False:
            return 10
        Thread.__init__(self)
        self.daemon = True
        self.tasks = tasks
        self.results = results
        self._run = True

    def abort(self):
        if False:
            return 10
        self._run = False

    def run(self):
        if False:
            i = 10
            return i + 15
        while self._run and (not self.tasks.empty()):
            try:
                (result, store_plugin, callback, timeout) = self.tasks.get()
                if result:
                    store_plugin.get_details(result, timeout)
                    callback(result)
                self.tasks.task_done()
            except:
                if DEBUG:
                    traceback.print_exc()

class CacheUpdateThreadPool(GenericDownloadThreadPool):

    def __init__(self, thread_count):
        if False:
            return 10
        GenericDownloadThreadPool.__init__(self, CacheUpdateThread, thread_count)

    def add_task(self, store_plugin, timeout=10):
        if False:
            print('Hello World!')
        self.tasks.put((store_plugin, timeout))
        GenericDownloadThreadPool.add_task(self)

class CacheUpdateThread(Thread):

    def __init__(self, tasks, results):
        if False:
            print('Hello World!')
        Thread.__init__(self)
        self.daemon = True
        self.tasks = tasks
        self._run = True

    def abort(self):
        if False:
            while True:
                i = 10
        self._run = False

    def run(self):
        if False:
            return 10
        while self._run and (not self.tasks.empty()):
            try:
                (store_plugin, timeout) = self.tasks.get()
                store_plugin.update_cache(timeout=timeout, suppress_progress=True)
            except:
                if DEBUG:
                    traceback.print_exc()