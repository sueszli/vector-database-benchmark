"""SCons.Job

This module defines the Serial and Parallel classes that execute tasks to
complete a build. The Jobs class provides a higher level interface to start,
stop, and wait on jobs.

"""
__revision__ = 'src/engine/SCons/Job.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.compat
import os
import signal
import SCons.Errors
explicit_stack_size = None
default_stack_size = 256
interrupt_msg = 'Build interrupted.'

class InterruptState(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.interrupted = False

    def set(self):
        if False:
            return 10
        self.interrupted = True

    def __call__(self):
        if False:
            return 10
        return self.interrupted

class Jobs(object):
    """An instance of this class initializes N jobs, and provides
    methods for starting, stopping, and waiting on all N jobs.
    """

    def __init__(self, num, taskmaster):
        if False:
            print('Hello World!')
        "\n        Create 'num' jobs using the given taskmaster.\n\n        If 'num' is 1 or less, then a serial job will be used,\n        otherwise a parallel job with 'num' worker threads will\n        be used.\n\n        The 'num_jobs' attribute will be set to the actual number of jobs\n        allocated.  If more than one job is requested but the Parallel\n        class can't do it, it gets reset to 1.  Wrapping interfaces that\n        care should check the value of 'num_jobs' after initialization.\n        "
        self.job = None
        if num > 1:
            stack_size = explicit_stack_size
            if stack_size is None:
                stack_size = default_stack_size
            try:
                self.job = Parallel(taskmaster, num, stack_size)
                self.num_jobs = num
            except NameError:
                pass
        if self.job is None:
            self.job = Serial(taskmaster)
            self.num_jobs = 1

    def run(self, postfunc=lambda : None):
        if False:
            i = 10
            return i + 15
        'Run the jobs.\n\n        postfunc() will be invoked after the jobs has run. It will be\n        invoked even if the jobs are interrupted by a keyboard\n        interrupt (well, in fact by a signal such as either SIGINT,\n        SIGTERM or SIGHUP). The execution of postfunc() is protected\n        against keyboard interrupts and is guaranteed to run to\n        completion.'
        self._setup_sig_handler()
        try:
            self.job.start()
        finally:
            postfunc()
            self._reset_sig_handler()

    def were_interrupted(self):
        if False:
            while True:
                i = 10
        'Returns whether the jobs were interrupted by a signal.'
        return self.job.interrupted()

    def _setup_sig_handler(self):
        if False:
            while True:
                i = 10
        "Setup an interrupt handler so that SCons can shutdown cleanly in\n        various conditions:\n\n          a) SIGINT: Keyboard interrupt\n          b) SIGTERM: kill or system shutdown\n          c) SIGHUP: Controlling shell exiting\n\n        We handle all of these cases by stopping the taskmaster. It\n        turns out that it's very difficult to stop the build process\n        by throwing asynchronously an exception such as\n        KeyboardInterrupt. For example, the python Condition\n        variables (threading.Condition) and queues do not seem to be\n        asynchronous-exception-safe. It would require adding a whole\n        bunch of try/finally block and except KeyboardInterrupt all\n        over the place.\n\n        Note also that we have to be careful to handle the case when\n        SCons forks before executing another process. In that case, we\n        want the child to exit immediately.\n        "

        def handler(signum, stack, self=self, parentpid=os.getpid()):
            if False:
                for i in range(10):
                    print('nop')
            if os.getpid() == parentpid:
                self.job.taskmaster.stop()
                self.job.interrupted.set()
            else:
                os._exit(2)
        self.old_sigint = signal.signal(signal.SIGINT, handler)
        self.old_sigterm = signal.signal(signal.SIGTERM, handler)
        try:
            self.old_sighup = signal.signal(signal.SIGHUP, handler)
        except AttributeError:
            pass

    def _reset_sig_handler(self):
        if False:
            i = 10
            return i + 15
        'Restore the signal handlers to their previous state (before the\n         call to _setup_sig_handler().'
        signal.signal(signal.SIGINT, self.old_sigint)
        signal.signal(signal.SIGTERM, self.old_sigterm)
        try:
            signal.signal(signal.SIGHUP, self.old_sighup)
        except AttributeError:
            pass

class Serial(object):
    """This class is used to execute tasks in series, and is more efficient
    than Parallel, but is only appropriate for non-parallel builds. Only
    one instance of this class should be in existence at a time.

    This class is not thread safe.
    """

    def __init__(self, taskmaster):
        if False:
            i = 10
            return i + 15
        "Create a new serial job given a taskmaster.\n\n        The taskmaster's next_task() method should return the next task\n        that needs to be executed, or None if there are no more tasks. The\n        taskmaster's executed() method will be called for each task when it\n        is successfully executed, or failed() will be called if it failed to\n        execute (e.g. execute() raised an exception)."
        self.taskmaster = taskmaster
        self.interrupted = InterruptState()

    def start(self):
        if False:
            return 10
        'Start the job. This will begin pulling tasks from the taskmaster\n        and executing them, and return when there are no more tasks. If a task\n        fails to execute (i.e. execute() raises an exception), then the job will\n        stop.'
        while True:
            task = self.taskmaster.next_task()
            if task is None:
                break
            try:
                task.prepare()
                if task.needs_execute():
                    task.execute()
            except Exception:
                if self.interrupted():
                    try:
                        raise SCons.Errors.BuildError(task.targets[0], errstr=interrupt_msg)
                    except:
                        task.exception_set()
                else:
                    task.exception_set()
                task.failed()
            else:
                task.executed()
            task.postprocess()
        self.taskmaster.cleanup()
try:
    import queue
    import threading
except ImportError:
    pass
else:

    class Worker(threading.Thread):
        """A worker thread waits on a task to be posted to its request queue,
        dequeues the task, executes it, and posts a tuple including the task
        and a boolean indicating whether the task executed successfully. """

        def __init__(self, requestQueue, resultsQueue, interrupted):
            if False:
                for i in range(10):
                    print('nop')
            threading.Thread.__init__(self)
            self.setDaemon(1)
            self.requestQueue = requestQueue
            self.resultsQueue = resultsQueue
            self.interrupted = interrupted
            self.start()

        def run(self):
            if False:
                for i in range(10):
                    print('nop')
            while True:
                task = self.requestQueue.get()
                if task is None:
                    break
                try:
                    if self.interrupted():
                        raise SCons.Errors.BuildError(task.targets[0], errstr=interrupt_msg)
                    task.execute()
                except:
                    task.exception_set()
                    ok = False
                else:
                    ok = True
                self.resultsQueue.put((task, ok))

    class ThreadPool(object):
        """This class is responsible for spawning and managing worker threads."""

        def __init__(self, num, stack_size, interrupted):
            if False:
                while True:
                    i = 10
            "Create the request and reply queues, and 'num' worker threads.\n\n            One must specify the stack size of the worker threads. The\n            stack size is specified in kilobytes.\n            "
            self.requestQueue = queue.Queue(0)
            self.resultsQueue = queue.Queue(0)
            try:
                prev_size = threading.stack_size(stack_size * 1024)
            except AttributeError as e:
                if explicit_stack_size is not None:
                    msg = 'Setting stack size is unsupported by this version of Python:\n    ' + e.args[0]
                    SCons.Warnings.warn(SCons.Warnings.StackSizeWarning, msg)
            except ValueError as e:
                msg = 'Setting stack size failed:\n    ' + str(e)
                SCons.Warnings.warn(SCons.Warnings.StackSizeWarning, msg)
            self.workers = []
            for _ in range(num):
                worker = Worker(self.requestQueue, self.resultsQueue, interrupted)
                self.workers.append(worker)
            if 'prev_size' in locals():
                threading.stack_size(prev_size)

        def put(self, task):
            if False:
                return 10
            'Put task into request queue.'
            self.requestQueue.put(task)

        def get(self):
            if False:
                for i in range(10):
                    print('nop')
            'Remove and return a result tuple from the results queue.'
            return self.resultsQueue.get()

        def preparation_failed(self, task):
            if False:
                for i in range(10):
                    print('nop')
            self.resultsQueue.put((task, False))

        def cleanup(self):
            if False:
                return 10
            '\n            Shuts down the thread pool, giving each worker thread a\n            chance to shut down gracefully.\n            '
            for _ in self.workers:
                self.requestQueue.put(None)
            for worker in self.workers:
                worker.join(1.0)
            self.workers = []

    class Parallel(object):
        """This class is used to execute tasks in parallel, and is somewhat
        less efficient than Serial, but is appropriate for parallel builds.

        This class is thread safe.
        """

        def __init__(self, taskmaster, num, stack_size):
            if False:
                for i in range(10):
                    print('nop')
            "Create a new parallel job given a taskmaster.\n\n            The taskmaster's next_task() method should return the next\n            task that needs to be executed, or None if there are no more\n            tasks. The taskmaster's executed() method will be called\n            for each task when it is successfully executed, or failed()\n            will be called if the task failed to execute (i.e. execute()\n            raised an exception).\n\n            Note: calls to taskmaster are serialized, but calls to\n            execute() on distinct tasks are not serialized, because\n            that is the whole point of parallel jobs: they can execute\n            multiple tasks simultaneously. "
            self.taskmaster = taskmaster
            self.interrupted = InterruptState()
            self.tp = ThreadPool(num, stack_size, self.interrupted)
            self.maxjobs = num

        def start(self):
            if False:
                print('Hello World!')
            'Start the job. This will begin pulling tasks from the\n            taskmaster and executing them, and return when there are no\n            more tasks. If a task fails to execute (i.e. execute() raises\n            an exception), then the job will stop.'
            jobs = 0
            while True:
                while jobs < self.maxjobs:
                    task = self.taskmaster.next_task()
                    if task is None:
                        break
                    try:
                        task.prepare()
                    except:
                        task.exception_set()
                        task.failed()
                        task.postprocess()
                    else:
                        if task.needs_execute():
                            self.tp.put(task)
                            jobs = jobs + 1
                        else:
                            task.executed()
                            task.postprocess()
                if not task and (not jobs):
                    break
                while True:
                    (task, ok) = self.tp.get()
                    jobs = jobs - 1
                    if ok:
                        task.executed()
                    else:
                        if self.interrupted():
                            try:
                                raise SCons.Errors.BuildError(task.targets[0], errstr=interrupt_msg)
                            except:
                                task.exception_set()
                        task.failed()
                    task.postprocess()
                    if self.tp.resultsQueue.empty():
                        break
            self.tp.cleanup()
            self.taskmaster.cleanup()