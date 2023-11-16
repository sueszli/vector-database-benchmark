from __future__ import print_function
from builtins import range
import os
import subprocess
import sys
import time
from multiprocessing import cpu_count, Queue, Process
from .test import Test

class Message(object):
    """Message exchanged in the TestSet message queue"""
    pass

class MessageTaskNew(Message):
    """Stand for a new task"""

    def __init__(self, task):
        if False:
            i = 10
            return i + 15
        self.task = task

class MessageTaskDone(Message):
    """Stand for a task done"""

    def __init__(self, task, error):
        if False:
            print('Hello World!')
        self.task = task
        self.error = error

class MessageClose(Message):
    """Close the channel"""
    pass

def worker(todo_queue, message_queue, init_args):
    if False:
        for i in range(10):
            print('nop')
    'Worker launched in parallel\n    @todo_queue: task to do\n    @message_queue: communication with Host\n    @init_args: additional arguments for command line\n    '
    while True:
        test = todo_queue.get()
        if test is None:
            break
        test.start_time = time.time()
        message_queue.put(MessageTaskNew(test))
        executable = test.executable if test.executable else sys.executable
        testpy = subprocess.Popen([executable] + init_args + test.command_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=test.base_dir)
        outputs = testpy.communicate()
        error = None
        if testpy.returncode != 0:
            error = outputs[1]
        message_queue.put(MessageTaskDone(test, error))

class TestSet(object):
    """Manage a set of test"""
    worker = staticmethod(worker)

    def __init__(self, base_dir):
        if False:
            for i in range(10):
                print('nop')
        'Initialise a test set\n        @base_dir: base directory for tests\n        '
        self.base_dir = base_dir
        self.task_done_cb = lambda tst, err: None
        self.task_new_cb = lambda tst: None
        self.todo_queue = Queue()
        self.message_queue = Queue()
        self.tests = []
        self.tests_done = []
        self.cpu_c = cpu_count()
        self.errorcode = 0
        self.additional_args = []

    def __add__(self, test):
        if False:
            i = 10
            return i + 15
        'Same as TestSet.add'
        self.add(test)
        return self

    def add(self, test):
        if False:
            print('Hello World!')
        'Add a test instance to the current test set'
        if not isinstance(test, Test):
            raise ValueError('%s is not a valid test instance' % repr(test))
        self.tests.append(test)

    def set_cpu_numbers(self, cpu_c):
        if False:
            for i in range(10):
                print('nop')
        'Set the number of cpu to use\n        @cpu_c: Number of CPU to use (default is maximum)\n        '
        self.cpu_c = cpu_c

    def set_callback(self, task_done=None, task_new=None):
        if False:
            while True:
                i = 10
        'Set callbacks for task information retrieval\n        @task_done: function(Test, Error message)\n        @task_new: function(Test)\n        '
        if task_done:
            self.task_done_cb = task_done
        if task_new:
            self.task_new_cb = task_new

    def _add_tasks(self):
        if False:
            i = 10
            return i + 15
        'Add tests to do, regarding to dependencies'
        for test in self.tests:
            launchable = True
            for dependency in test.depends:
                if dependency not in self.tests_done:
                    launchable = False
                    break
            if launchable:
                self.tests.remove(test)
                self.todo_queue.put(test)
        if len(self.tests) == 0:
            for _ in range(self.cpu_c):
                self.todo_queue.put(None)
        if len(self.tests_done) == self.init_tests_number:
            self.message_queue.put(MessageClose())

    def _messages_handler(self):
        if False:
            i = 10
            return i + 15
        'Manage message between Master and Workers'
        while True:
            message = self.message_queue.get()
            if isinstance(message, MessageClose):
                break
            elif isinstance(message, MessageTaskNew):
                self.task_new_cb(message.task)
            elif isinstance(message, MessageTaskDone):
                self.tests_done.append(message.task)
                self._add_tasks()
                self.task_done_cb(message.task, message.error)
                if message.error is not None:
                    self.errorcode = -1
            else:
                raise ValueError('Unknown message type %s' % type(message))

    @staticmethod
    def fast_unify(seq, idfun=None):
        if False:
            print('Hello World!')
        'Order preserving unifying list function\n        @seq: list to unify\n        @idfun: marker function (default is identity)\n        '
        if idfun is None:
            idfun = lambda x: x
        seen = {}
        result = []
        for item in seq:
            marker = idfun(item)
            if marker in seen:
                continue
            seen[marker] = 1
            result.append(item)
        return result

    def _clean(self):
        if False:
            while True:
                i = 10
        'Remove produced files'
        products = []
        current_directory = os.getcwd()
        for test in self.tests_done:
            for product in test.products:
                products.append(os.path.join(current_directory, test.base_dir, product))
        for product in TestSet.fast_unify(products):
            try:
                os.remove(product)
            except OSError:
                print('Cleaning error: Unable to remove %s' % product)

    def add_additional_args(self, args):
        if False:
            print('Hello World!')
        'Add arguments to used on the test command line\n        @args: list of str\n        '
        self.additional_args += args

    def run(self):
        if False:
            while True:
                i = 10
        'Launch tests'
        self.current_directory = os.getcwd()
        os.chdir(self.base_dir)
        processes = []
        for _ in range(self.cpu_c):
            p = Process(target=TestSet.worker, args=(self.todo_queue, self.message_queue, self.additional_args))
            processes.append(p)
            p.start()
        self.init_tests_number = len(self.tests)
        self._add_tasks()
        self._messages_handler()
        self.todo_queue.close()
        self.todo_queue.join_thread()
        self.message_queue.close()
        self.message_queue.join_thread()
        for p in processes:
            p.join()

    def end(self, clean=True):
        if False:
            print('Hello World!')
        'End a testset run\n        @clean: (optional) if set, remove tests products\n        PRE: run()\n        '
        if clean:
            self._clean()
        os.chdir(self.current_directory)

    def tests_passed(self):
        if False:
            while True:
                i = 10
        'Return a non zero value if at least one test failed'
        return self.errorcode

    def filter_tags(self, include_tags=None, exclude_tags=None):
        if False:
            while True:
                i = 10
        "Filter tests by tags\n        @include_tags: list of tags' name (whitelist)\n        @exclude_tags: list of tags' name (blacklist)\n        If @include_tags and @exclude_tags are used together, @exclude_tags will\n        act as a blacklist on @include_tags generated tests\n        "
        new_testset = []
        include_tags = set(include_tags)
        exclude_tags = set(exclude_tags)
        if include_tags.intersection(exclude_tags):
            raise ValueError('Tags are mutually included and excluded: %s' % include_tags.intersection(exclude_tags))
        for test in self.tests:
            tags = set(test.tags)
            if exclude_tags.intersection(tags):
                continue
            if not include_tags:
                new_testset.append(test)
            elif include_tags.intersection(tags):
                new_testset.append(test)
                dependency = list(test.depends)
                while dependency:
                    subtest = dependency.pop()
                    if subtest not in new_testset:
                        new_testset.append(subtest)
                    for subdepends in subtest.depends:
                        if subdepends not in new_testset:
                            dependency.append(subdepends)
        self.tests = new_testset