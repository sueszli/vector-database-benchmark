"""
A module containing classes used to simulate certain behaviors
"""
from multiprocessing import Value
import tempfile
import hashlib
import logging
import os
import luigi
logger = logging.getLogger('luigi-interface')

class RunAnywayTarget(luigi.Target):
    """
    A target used to make a task run every time it is called.

    Usage:

    Pass `self` as the first argument in your task's `output`:

    .. code-block: python

        def output(self):
            return RunAnywayTarget(self)

    And then mark it as `done` in your task's `run`:

    .. code-block: python

        def run(self):
            # Your task execution
            # ...
            self.output().done() # will then be considered as "existing"
    """
    temp_dir = os.path.join(tempfile.gettempdir(), 'luigi-simulate')
    temp_time = 24 * 3600
    unique = Value('i', 0)

    def __init__(self, task_obj):
        if False:
            for i in range(10):
                print('nop')
        self.task_id = task_obj.task_id
        if self.unique.value == 0:
            with self.unique.get_lock():
                if self.unique.value == 0:
                    self.unique.value = os.getpid()
        if os.path.isdir(self.temp_dir):
            import shutil
            import time
            limit = time.time() - self.temp_time
            for fn in os.listdir(self.temp_dir):
                path = os.path.join(self.temp_dir, fn)
                if os.path.isdir(path) and os.stat(path).st_mtime < limit:
                    shutil.rmtree(path)
                    logger.debug('Deleted temporary directory %s', path)

    def __str__(self):
        if False:
            print('Hello World!')
        return self.task_id

    def get_path(self):
        if False:
            print('Hello World!')
        "\n        Returns a temporary file path based on a MD5 hash generated with the task's name and its arguments\n        "
        md5_hash = hashlib.new('md5', self.task_id.encode(), usedforsecurity=False).hexdigest()
        logger.debug('Hash %s corresponds to task %s', md5_hash, self.task_id)
        return os.path.join(self.temp_dir, str(self.unique.value), md5_hash)

    def exists(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if the file exists\n        '
        return os.path.isfile(self.get_path())

    def done(self):
        if False:
            while True:
                i = 10
        '\n        Creates temporary file to mark the task as `done`\n        '
        logger.info('Marking %s as done', self)
        fn = self.get_path()
        try:
            os.makedirs(os.path.dirname(fn))
        except OSError:
            pass
        open(fn, 'w').close()