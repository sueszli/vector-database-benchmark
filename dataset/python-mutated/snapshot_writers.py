import multiprocessing
import os
import shutil
import threading
from six.moves import queue
from chainer.serializers import npz
from chainer import utils

class Writer(object):
    """Base class of snapshot writers.

    :class:`~chainer.training.extensions.Snapshot` invokes ``__call__`` of this
    class every time when taking a snapshot.
    This class determines how the actual saving function will be invoked.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self._post_save_hooks = []

    def __call__(self, filename, outdir, target):
        if False:
            i = 10
            return i + 15
        'Invokes the actual snapshot function.\n\n        This method is invoked by a\n        :class:`~chainer.training.extensions.Snapshot` object every time it\n        takes a snapshot.\n\n        Args:\n            filename (str): Name of the file into which the serialized target\n                is saved. It is a concrete file name, i.e. not a pre-formatted\n                template string.\n            outdir (str): Output directory. Corresponds to\n                :py:attr:`Trainer.out <chainer.training.Trainer.out>`.\n            target (dict): Serialized object which will be saved.\n        '
        raise NotImplementedError

    def __del__(self):
        if False:
            print('Hello World!')
        self.finalize()

    def finalize(self):
        if False:
            while True:
                i = 10
        'Finalizes the wirter.\n\n        Like extensions in :class:`~chainer.training.Trainer`, this method\n        is invoked at the end of the training.\n\n        '
        pass

    def save(self, filename, outdir, target, savefun, **kwds):
        if False:
            print('Hello World!')
        prefix = 'tmp' + filename
        with utils.tempdir(prefix=prefix, dir=outdir) as tmpdir:
            tmppath = os.path.join(tmpdir, filename)
            savefun(tmppath, target)
            shutil.move(tmppath, os.path.join(outdir, filename))
        self._post_save()

    def _add_cleanup_hook(self, hook_fun):
        if False:
            for i in range(10):
                print('nop')
        'Adds cleanup hook function.\n\n        Technically, arbitrary user-defined hook can be called, but\n        this is intended for cleaning up stale snapshots.\n\n        Args:\n            hook_fun (callable): callable function to be called\n                right after save is done. It takes no arguments.\n\n        '
        self._post_save_hooks.append(hook_fun)

    def _post_save(self):
        if False:
            while True:
                i = 10
        for hook in self._post_save_hooks:
            hook()

class SimpleWriter(Writer):
    """The most simple snapshot writer.

    This class just passes the arguments to the actual saving function.

    Args:
        savefun: Callable object. It takes three arguments: the output file
            path, the serialized dictionary object, and the optional keyword
            arguments.
        kwds: Keyword arguments for the ``savefun``.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    def __init__(self, savefun=npz.save_npz, **kwds):
        if False:
            i = 10
            return i + 15
        super(SimpleWriter, self).__init__()
        self._savefun = savefun
        self._kwds = kwds

    def __call__(self, filename, outdir, target):
        if False:
            i = 10
            return i + 15
        self.save(filename, outdir, target, self._savefun, **self._kwds)

class StandardWriter(Writer):
    """Base class of snapshot writers which use thread or process.

    This class creates a new thread or a process every time when ``__call__``
    is invoked.

    Args:
        savefun: Callable object. It takes three arguments: the output file
            path, the serialized dictionary object, and the optional keyword
            arguments.
        kwds: Keyword arguments for the ``savefun``.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """
    _started = False
    _finalized = False
    _worker = None

    def __init__(self, savefun=npz.save_npz, **kwds):
        if False:
            while True:
                i = 10
        super(StandardWriter, self).__init__()
        self._savefun = savefun
        self._kwds = kwds
        self._started = False
        self._finalized = False

    def __call__(self, filename, outdir, target):
        if False:
            return 10
        if self._started:
            self._worker.join()
            self._started = False
        self._filename = filename
        self._worker = self.create_worker(filename, outdir, target, **self._kwds)
        self._worker.start()
        self._started = True

    def create_worker(self, filename, outdir, target, **kwds):
        if False:
            i = 10
            return i + 15
        'Creates a worker for the snapshot.\n\n        This method creates a thread or a process to take a snapshot. The\n        created worker must have :meth:`start` and :meth:`join` methods.\n\n        Args:\n            filename (str): Name of the file into which the serialized target\n                is saved. It is already formated string.\n            outdir (str): Output directory. Passed by `trainer.out`.\n            target (dict): Serialized object which will be saved.\n            kwds: Keyword arguments for the ``savefun``.\n\n        '
        raise NotImplementedError

    def finalize(self):
        if False:
            i = 10
            return i + 15
        if self._started:
            if not self._finalized:
                self._worker.join()
            self._started = False
        self._finalized = True

class ThreadWriter(StandardWriter):
    """Snapshot writer that uses a separate thread.

    This class creates a new thread that invokes the actual saving function.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    def __init__(self, savefun=npz.save_npz, **kwds):
        if False:
            for i in range(10):
                print('nop')
        super(ThreadWriter, self).__init__(savefun=savefun, **kwds)

    def create_worker(self, filename, outdir, target, **kwds):
        if False:
            while True:
                i = 10
        return threading.Thread(target=self.save, args=(filename, outdir, target, self._savefun), kwargs=self._kwds)

class ProcessWriter(StandardWriter):
    """Snapshot writer that uses a separate process.

    This class creates a new process that invokes the actual saving function.

    .. note::
        Forking a new process from a MPI process might be danger. Consider
        using :class:`ThreadWriter` instead of ``ProcessWriter`` if you are
        using MPI.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    def __init__(self, savefun=npz.save_npz, **kwds):
        if False:
            print('Hello World!')
        super(ProcessWriter, self).__init__(savefun=savefun, **kwds)

    def create_worker(self, filename, outdir, target, **kwds):
        if False:
            for i in range(10):
                print('nop')
        return multiprocessing.Process(target=self.save, args=(filename, outdir, target, self._savefun), kwargs=self._kwds)

class QueueWriter(Writer):
    """Base class of queue snapshot writers.

    This class is a base class of snapshot writers that use a queue.
    A Queue is created when this class is constructed, and every time when
    ``__call__`` is invoked, a snapshot task is put into the queue.

    Args:
        savefun: Callable object which is passed to the :meth:`create_task`
            if the task is ``None``. It takes three arguments: the output file
            path, the serialized dictionary object, and the optional keyword
            arguments.
        task: Callable object. Its ``__call__`` must have a same interface to
            ``Writer.__call__``. This object is directly put into the queue.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """
    _started = False
    _finalized = False
    _queue = None
    _consumer = None

    def __init__(self, savefun=npz.save_npz, task=None):
        if False:
            return 10
        super(QueueWriter, self).__init__()
        if task is None:
            self._task = self.create_task(savefun)
        else:
            self._task = task
        self._queue = self.create_queue()
        self._consumer = self.create_consumer(self._queue)
        self._consumer.start()
        self._started = True
        self._finalized = False

    def __call__(self, filename, outdir, target):
        if False:
            print('Hello World!')
        self._queue.put([self._task, filename, outdir, target])

    def create_task(self, savefun):
        if False:
            print('Hello World!')
        return SimpleWriter(savefun=savefun)

    def create_queue(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def create_consumer(self, q):
        if False:
            return 10
        raise NotImplementedError

    def consume(self, q):
        if False:
            for i in range(10):
                print('nop')
        while True:
            task = q.get()
            if task is None:
                q.task_done()
                return
            else:
                task[0](task[1], task[2], task[3])
                q.task_done()

    def finalize(self):
        if False:
            return 10
        if self._started:
            if not self._finalized:
                self._queue.put(None)
                self._queue.join()
                self._consumer.join()
            self._started = False
        self._finalized = True

class ThreadQueueWriter(QueueWriter):
    """Snapshot writer that uses a thread queue.

    This class creates a thread and a queue by :mod:`threading` and
    :mod:`queue` modules
    respectively. The thread will be a consumer of the queue, and the main
    thread will be a producer of the queue.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    def __init__(self, savefun=npz.save_npz, task=None):
        if False:
            return 10
        super(ThreadQueueWriter, self).__init__(savefun=savefun, task=task)

    def create_queue(self):
        if False:
            return 10
        return queue.Queue()

    def create_consumer(self, q):
        if False:
            i = 10
            return i + 15
        return threading.Thread(target=self.consume, args=(q,))

class ProcessQueueWriter(QueueWriter):
    """Snapshot writer that uses process queue.

    This class creates a process and a queue by :mod:`multiprocessing` module.
    The process will be a consumer of this queue, and the main process will be
    a producer of this queue.

    .. note::
        Forking a new process from MPI process might be danger. Consider using
        :class:`ThreadQueueWriter` instead of ``ProcessQueueWriter`` if you are
        using MPI.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    def __init__(self, savefun=npz.save_npz, task=None):
        if False:
            for i in range(10):
                print('nop')
        super(ProcessQueueWriter, self).__init__(savefun=savefun, task=task)

    def create_queue(self):
        if False:
            while True:
                i = 10
        return multiprocessing.JoinableQueue()

    def create_consumer(self, q):
        if False:
            for i in range(10):
                print('nop')
        return multiprocessing.Process(target=self.consume, args=(q,))