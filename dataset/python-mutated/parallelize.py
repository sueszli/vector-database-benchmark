from __future__ import print_function
import threading
import queue
import sys
import os
import pickle
import traceback
import time
import signal
import multiprocessing
import vaex.misc.progressbar
try:
    from io import StringIO
except ImportError:
    from cStringIO import StringIO
pickle_encoding = 'hex'
pickle_protocol = 2

def log(*args):
    if False:
        for i in range(10):
            print('nop')
    pass

def serialize(obj):
    if False:
        return 10
    data = pickle.dumps(obj, pickle_protocol)
    rawdata = data.encode(pickle_encoding)
    return rawdata

def deserialize(rawdata):
    if False:
        for i in range(10):
            print('nop')
    data = rawdata.decode(pickle_encoding)
    obj = pickle.loads(data)
    return obj

class InfoThread(threading.Thread):

    def __init__(self, fullsize, executions):
        if False:
            print('Hello World!')
        threading.Thread.__init__(self)
        self.fullsize = fullsize
        self.executions = executions
        self.setDaemon(True)

    def n_done(self):
        if False:
            while True:
                i = 10
        n = 0
        for execution in self.executions:
            n += len(execution.results)
        return n

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        while 1:
            count = self.n_done()
            print('%d out of %d tasks completed (%5.2f%%)' % (count, self.fullsize, float(count) / self.fullsize * 100))
            time.sleep(0.1)

class Watchdog(threading.Thread):

    def __init__(self, executions):
        if False:
            i = 10
            return i + 15
        threading.Thread.__init__(self)
        self.executions = executions
        self.setDaemon(True)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        while 1:
            time.sleep(0.5)
            for execution in self.executions:
                print(execution.isAlive(), execution.error)

class InfoThreadProgressBar(InfoThread):

    def __init__(self, fullsize, executions):
        if False:
            return 10
        InfoThread.__init__(self, fullsize, executions)
        self.bar = vaex.misc.progressbar.ProgressBar(0, fullsize)

    def run(self):
        if False:
            i = 10
            return i + 15
        done = False
        error = False
        while not done and (not error):
            count = self.n_done()
            self.bar.update(count)
            time.sleep(0.1)
            done = count == self.fullsize
            for execution in self.executions:
                if execution.error:
                    error = True

class LocalExecutor(object):

    def __init__(self, thread, function):
        if False:
            for i in range(10):
                print('nop')
        self.thread = thread
        self.function = function

    def stop(self):
        if False:
            print('Hello World!')
        pass

    def init(self):
        if False:
            print('Hello World!')
        pass

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        info = 'ok'
        exc_info = None
        result = None
        try:
            result = self.function(*args, **kwargs)
        except:
            info = 'exception'
            exc_info = traceback.format_exc()
        return (info, exc_info, result)

class IOExecutor(object):

    def __init__(self, thread, function, init, id_number):
        if False:
            return 10
        self.thread = thread
        self.function = function
        self.initf = init
        self.id_number = id_number
        self.createIO()
        self.error = False

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.log('p: sending execute command')
        self.output.write('execute\n')
        self.output.write(serialize((args, kwargs)))
        self.output.write('\n')
        self.output.flush()
        self.log('p: reading response')
        response = self.input.readline().strip()
        (error, exc_info, result) = deserialize(response)
        return (error, exc_info, result)

    def init(self):
        if False:
            print('Hello World!')
        return True

    def stop(self, error=False):
        if False:
            i = 10
            return i + 15
        self.error = error
        self.log('p: sending stop command')
        self.output.write('stop\n')
        self.output.flush()
        self.log('waitpid')
        self.join()
        self.log('waitpid done')
        self.input.close()
        self.output.close()

    def join(self):
        if False:
            return 10
        os.waitpid(self.pid, 0)

    def log(self, *args):
        if False:
            for i in range(10):
                print('nop')
        log(self.thread.getName() + ':', ' '.join([str(k) for k in args]))

class ForkExecutor(IOExecutor):

    def createIO(self):
        if False:
            for i in range(10):
                print('nop')
        self.log('forking')
        (self.parent_input_p, self.child_output_p) = os.pipe()
        (self.child_input_p, self.parent_output_p) = os.pipe()
        self.pid = os.fork()
        if not self.pid:
            self.child_output = os.fdopen(self.child_output_p, 'w')
            self.child_input = os.fdopen(self.child_input_p, 'r')
            self.output = self.child_output
            self.input = self.child_input
            os.close(self.parent_output_p)
            os.close(self.parent_input_p)
            try:
                self.childPart()
            except BaseException:
                print('oops')
            self.input.close()
            self.output.close()
            os._exit(0)
        else:
            self.parent_output = os.fdopen(self.parent_output_p, 'w')
            self.parent_input = os.fdopen(self.parent_input_p, 'r')
            self.output = self.parent_output
            self.input = self.parent_input
            os.close(self.child_output_p)
            os.close(self.child_input_p)

    def childPart(self):
        if False:
            print('Hello World!')
        done = False
        self.log('c: child')
        if self.initf is not None:
            self.initf(self.id_number)
        while not done:
            self.log('c: waiting for command...')
            command = self.input.readline().strip()
            self.log('command:', command)
            if command == 'execute':
                response = self.input.readline().strip()
                self.log('c: args:', repr(response))
                (args, kwargs) = deserialize(response)
                info = 'ok'
                exc_info = None
                result = None
                try:
                    result = self.function(*args, **kwargs)
                except BaseException:
                    info = 'exception'
                    exc_info = traceback.format_exc()
                except KeyboardInterrupt:
                    info = 'exception'
                    exc_info = traceback.format_exc()
                self.log('c: pickling')
                self.output.write(serialize((info, exc_info, result)))
                self.output.write('\n')
                self.output.flush()
                self.log('c: closing')
            elif command == 'stop':
                self.log('c: stopping...')
                done = True
                self.log('c: closed, exiting')
            else:
                done = True
                self.log('c: unknown command', repr(command))

    def join(self):
        if False:
            while True:
                i = 10
        os.waitpid(self.pid, 0)

class Execution(threading.Thread):

    def __init__(self, taskQueue, fork, function, init, id_number, args, kwargs):
        if False:
            return 10
        self.taskQueue = taskQueue
        self.fork = fork
        self.function = function
        self.init = init
        self.id_number = id_number
        self.args = args
        self.kwargs = kwargs
        self.results = []
        self.done = False
        self.error = False
        threading.Thread.__init__(self)
        if self.fork:
            self.executor = ForkExecutor(self, self.function, self.init, self.id_number)
        else:
            self.executor = LocalExecutor(self, self.function)

    def log(self, *args):
        if False:
            for i in range(10):
                print('nop')
        log(self.getName() + ':', ' '.join([str(k) for k in args]))

    def run(self):
        if False:
            i = 10
            return i + 15
        task = None
        if self.executor.init():
            task = None
            self.log('starting')
            try:
                task = self.taskQueue.get(False)
            except queue.Empty:
                self.log('empty at first try')
            while task is not None and self.error is False:
                (tasknr, args) = task
                args = list(args)
                common_args = list(self.args)
                args.extend(common_args)
                kwargs = dict(self.kwargs)
                (info, exc_info, result) = self.executor(*args, **kwargs)
                if info == 'exception':
                    print(exc_info)
                    self.error = True
                self.log('r: got result')
                self.results.append((tasknr, result))
                try:
                    task = self.taskQueue.get(False)
                except queue.Empty:
                    self.log('empty queue')
                    task = None
            self.executor.stop(self.error)
            self.log('done')
        else:
            self.log('failure starting slave')
            self.error = True
        self.done = True

def countcores():
    if False:
        print('Hello World!')
    (stdin, stdout) = os.popen2('cat /proc/cpuinfo | grep processor')
    lines = stdout.readlines()
    return len(lines)

def timed(f):
    if False:
        for i in range(10):
            print('nop')

    def execute(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        t0 = time.time()
        (utime0, stime0, child_utime0, child_stime0, walltime0) = os.times()
        result = f(*args, **kwargs)
        dt = time.time() - t0
        (utime, stime, child_utime, child_stime, walltime) = os.times()
        print()
        print('user time:            % 9.3f sec.' % (utime - utime0))
        print('system time:          % 9.3f sec.' % (stime - stime0))
        print('user time(children):  % 9.3f sec.' % (child_utime - child_utime0))
        print('system time(children):% 9.3f sec.' % (child_stime - child_stime0))
        print()
        dt_total = child_utime - child_utime0 + child_stime - child_stime0 + utime - utime0 + stime - stime0
        print('total cpu time:       % 9.3f sec. (time it would take on a single cpu/core)' % dt_total)
        print('elapsed time:         % 9.3f sec. (normal wallclock time it took)' % (walltime - walltime0))
        dt = walltime - walltime0
        if dt == 0:
            eff = 0.0
        else:
            eff = dt_total / dt
        print('efficiency factor     % 9.3f      (ratio of the two above ~= # cores)' % eff)
        return result
    return execute

def parallelize(cores=None, fork=True, flatten=False, info=False, infoclass=InfoThreadProgressBar, init=None, *args, **kwargs):
    if False:
        while True:
            i = 10
    'Function decorater that executes the function in parallel\n\n    Usage::\n\n            @parallelize(cores=10, info=True)\n            def f(x):\n                    return x**2\n\n            x = numpy.arange(0, 100, 0.1)\n            y = f(x) # this gets executed parallel\n\n    :param cores: number of cpus/cores to use (if None, it counts the cores using /proc/cpuinfo)\n    :param fork: fork a process (should always be true since of the GIT, but can be used with c modules that release the GIT)\n    :param flatten: if False and each return value is a list, final result will be a list of lists, if True, all lists are combined to one big list\n    :param info: show progress bar (see infoclass)\n    :param infoclass: class to instantiate that shows the progress (default shows progressbar)\n    :param init: function to be called in each forked process before executing, can be used to set the seed, takes a integer as parameter (number that identifies the process)\n    :param args: extra arguments passed to function\n    :param kwargs: extra keyword arguments passed to function\n\n    Example::\n\n            @parallelize(cores=10, info=True, n=2)\n            def f(x, n):\n                    return x**n\n\n            x = numpy.arange(0, 100, 0.1)\n            y = f(x) # this gets executed parallel\n\n\n\n    '
    if cores is None:
        cores = multiprocessing.cpu_count()

    def wrapper(f):
        if False:
            for i in range(10):
                print('nop')

        def execute(*multiargs):
            if False:
                while True:
                    i = 10
            results = []
            len(list(zip(*multiargs)))
            N = len(multiargs[0])
            if info:
                print('running %i jobs on %i cores' % (N, cores))
            taskQueue = queue.Queue(len(multiargs[0]))
            for (tasknr, _args) in enumerate(zip(*multiargs)):
                taskQueue.put((tasknr, list(_args)))
            executions = [Execution(taskQueue, fork, f, init, corenr, args, kwargs) for corenr in range(cores)]
            if info:
                infoobj = infoclass(len(multiargs[0]), executions)
                infoobj.start()
            for (i, execution) in enumerate(executions):
                execution.setName('T-%d' % i)
                execution.start()
            error = False
            for execution in executions:
                log('joining:', execution.getName())
                try:
                    execution.join()
                except BaseException:
                    error = True
                results.extend(execution.results)
                if execution.error:
                    error = True
            if info:
                infoobj.join()
            if error:
                print('error', file=sys.stderr)
                results = None
                raise Exception('error in one or more of the executors')
            else:
                results.sort(cmp=lambda a, b: cmp(a[0], b[0]))
                results = [k[1] for k in results]
                if flatten:
                    flatresults = []
                    for result in results:
                        flatresults.extend(result)
                    results = flatresults
            return results
        return execute
    return wrapper
if __name__ == '__main__':

    @timed
    @parallelize(cores=6, fork=True, flatten=True, text='common argument')
    def testprime(from_nr, to_nr, text=''):
        if False:
            while True:
                i = 10
        primes = []
        from_nr = max(from_nr, 2)
        for p in range(from_nr, to_nr):
            isprime = True
            time.sleep(1)
            for i in range(2, p):
                if p % i == 0:
                    isprime = False
                    break
            if isprime:
                primes.append(p)
        return primes
    testnumbers = list(range(0, 10001, 100))
    from_nrs = testnumbers[:-1]
    to_nrs = testnumbers[1:]
    results = testprime(from_nrs, to_nrs)