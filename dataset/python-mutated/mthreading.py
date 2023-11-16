"""
Anything that has to do with threading in this library
must be abstracted in this file. If we decide to do gevent
also, it will deserve its own gevent file.
"""
__title__ = 'newspaper'
__author__ = 'Lucas Ou-Yang'
__license__ = 'MIT'
__copyright__ = 'Copyright 2014, Lucas Ou-Yang'
import logging
import queue
import traceback
from threading import Thread
from .configuration import Configuration
log = logging.getLogger(__name__)

class ConcurrencyException(Exception):
    pass

class Worker(Thread):
    """
    Thread executing tasks from a given tasks queue.
    """

    def __init__(self, tasks, timeout_seconds):
        if False:
            for i in range(10):
                print('nop')
        Thread.__init__(self)
        self.tasks = tasks
        self.timeout = timeout_seconds
        self.daemon = True
        self.start()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        while True:
            try:
                (func, args, kargs) = self.tasks.get(timeout=self.timeout)
            except queue.Empty:
                break
            try:
                func(*args, **kargs)
            except Exception:
                traceback.print_exc()
            self.tasks.task_done()

class ThreadPool:

    def __init__(self, num_threads, timeout_seconds):
        if False:
            i = 10
            return i + 15
        self.tasks = queue.Queue(num_threads)
        for _ in range(num_threads):
            Worker(self.tasks, timeout_seconds)

    def add_task(self, func, *args, **kargs):
        if False:
            print('Hello World!')
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        if False:
            i = 10
            return i + 15
        self.tasks.join()

class NewsPool(object):

    def __init__(self, config=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Abstraction of a threadpool. A newspool can accept any number of\n        source OR article objects together in a list. It allocates one\n        thread to every source and then joins.\n\n        We allocate one thread per source to avoid rate limiting.\n        5 sources = 5 threads, one per source.\n\n        >>> import newspaper\n        >>> from newspaper import news_pool\n\n        >>> cnn_paper = newspaper.build('http://cnn.com')\n        >>> tc_paper = newspaper.build('http://techcrunch.com')\n        >>> espn_paper = newspaper.build('http://espn.com')\n\n        >>> papers = [cnn_paper, tc_paper, espn_paper]\n        >>> news_pool.set(papers)\n        >>> news_pool.join()\n\n        # All of your papers should have their articles html all populated now.\n        >>> cnn_paper.articles[50].html\n        u'<html>blahblah ... '\n        "
        self.pool = None
        self.config = config or Configuration()

    def join(self):
        if False:
            while True:
                i = 10
        '\n        Runs the mtheading and returns when all threads have joined\n        resets the task.\n        '
        if self.pool is None:
            raise ConcurrencyException('Call set(..) with a list of source objects before calling .join(..)')
        self.pool.wait_completion()
        self.pool = None

    def set(self, news_list, threads_per_source=1, override_threads=None):
        if False:
            while True:
                i = 10
        '\n        news_list can be a list of `Article`, `Source`, or both.\n\n        If caller wants to decide how many threads to use, they can use\n        `override_threads` which takes precedence over all. Otherwise,\n        this api infers that if the input is all `Source` objects, to\n        allocate one thread per `Source` to not spam the host.\n\n        If both of the above conditions are not true, default to 1 thread.\n        '
        from .source import Source
        if override_threads is not None:
            num_threads = override_threads
        elif all([isinstance(n, Source) for n in news_list]):
            num_threads = threads_per_source * len(news_list)
        else:
            num_threads = 1
        timeout = self.config.thread_timeout_seconds
        self.pool = ThreadPool(num_threads, timeout)
        for news_object in news_list:
            if isinstance(news_object, Source):
                self.pool.add_task(news_object.download_articles)
            else:
                self.pool.add_task(news_object.download)