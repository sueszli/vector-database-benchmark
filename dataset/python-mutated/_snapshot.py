import os
import warnings
import chainer
from chainer.serializers import npz
from chainer.training import extension
from chainer.training.extensions import snapshot_writers
from chainer.utils import argument

def _find_snapshot_files(fmt, path):
    if False:
        print('Hello World!')
    'Only prefix and suffix match\n\n    TODO(kuenishi): currently clean format string such as\n    "snapshot{.iteration}.npz" can only be parsed, but tricky (or\n    invalid) formats like "snapshot{{.iteration}}.npz" are hard to\n    detect and to properly show errors, just ignored or fails so far.\n\n    Args:\n        fmt (str): format string to match with file names of\n            existing snapshots, where prefix and suffix are\n            only examined. Also, files\' staleness is judged\n            by timestamps. The default is metime.\n        path (str): a directory path to search for snapshot files.\n\n    Returns:\n        A sorted list of pair of ``mtime, filename``, whose file\n        name that matched the format ``fmt`` directly under ``path``.\n\n    '
    prefix = fmt.split('{')[0]
    suffix = fmt.split('}')[-1]
    matched_files = (file for file in os.listdir(path) if file.startswith(prefix) and file.endswith(suffix))

    def _prepend_mtime(f):
        if False:
            return 10
        t = os.stat(os.path.join(path, f)).st_mtime
        return (t, f)
    return sorted((_prepend_mtime(file) for file in matched_files))

def _find_latest_snapshot(fmt, path):
    if False:
        i = 10
        return i + 15
    "Finds the latest snapshots in a directory\n\n    Args:\n        fmt (str): format string to match with file names of\n            existing snapshots, where prefix and suffix are\n            only examined. Also, files' staleness is judged\n            by timestamps. The default is metime.\n        path (str): a directory path to search for snapshot files.\n\n    Returns:\n        Latest snapshot file, in terms of a file that has newest\n        ``mtime`` that matches format ``fmt`` directly under\n        ``path``. If no such file found, it returns ``None``.\n\n    "
    snapshot_files = _find_snapshot_files(fmt, path)
    if len(snapshot_files) > 0:
        (_, filename) = snapshot_files[-1]
        return filename
    return None

def _find_stale_snapshots(fmt, path, n_retains, **kwargs):
    if False:
        return 10
    "Finds stale snapshots in a directory, retaining several files\n\n    Args:\n        fmt (str): format string to match with file names of\n            existing snapshots, where prefix and suffix are\n            only examined. Also, files' staleness is judged\n            by timestamps. The default is metime.\n        path (str): a directory path to search for snapshot files.\n        n_retains (int): Number of snapshot files to retain\n            through the cleanup. Must be a positive integer for any cleanup to\n            take place.\n        num_retain (int): Same as ``n_retains`` (deprecated).\n\n    Yields:\n        str: The next stale file that matches format\n        ``fmt`` directly under ``path`` and with older ``mtime``,\n        excluding newest ``n_retains`` files.\n\n    "
    if 'num_retain' in kwargs:
        warnings.warn('Argument `num_retain` is deprecated. Please use `n_retains` instead', DeprecationWarning)
        n_retains = kwargs['num_retain']
    snapshot_files = _find_snapshot_files(fmt, path)
    n_removes = len(snapshot_files) - n_retains
    if n_removes > 0:
        for (_, filename) in snapshot_files[:n_removes]:
            yield filename

def snapshot_object(target, filename, savefun=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "snapshot_object(target, filename, savefun=None, *, condition=None, writer=None, snapshot_on_error=False, n_retains=-1, autoload=False)\n\n    Returns a trainer extension to take snapshots of a given object.\n\n    This extension serializes the given object and saves it to the output\n    directory.\n\n    This extension is called once per epoch by default. To take a\n    snapshot at a different interval, a trigger object specifying the\n    required interval can be passed along with this extension\n    to the `extend()` method of the trainer.\n\n    The default priority is -100, which is lower than that of most\n    built-in extensions.\n\n    Args:\n        target: Object to serialize.\n        filename (str): Name of the file into which the object is serialized.\n            It can be a format string, where the trainer object is passed to\n            the :meth:`str.format` method. For example,\n            ``'snapshot_{.updater.iteration}'`` is converted to\n            ``'snapshot_10000'`` at the 10,000th iteration.\n        savefun: Function to save the object. It takes two arguments: the\n            output file path and the object to serialize.\n        condition: Condition object. It must be a callable object that returns\n            boolean without any arguments. If it returns ``True``, the snapshot\n            will be done.\n            If not, it will be skipped. The default is a function that always\n            returns ``True``.\n        writer: Writer object.\n            It must be a callable object.\n            See below for the list of built-in writers.\n            If ``savefun`` is other than ``None``, this argument must be\n            ``None``. In that case, a\n            :class:`~chainer.training.extensions.snapshot_writers.SimpleWriter`\n            object instantiated with specified ``savefun`` argument will be\n            used.\n        snapshot_on_error (bool): Whether to take a snapshot in case trainer\n            loop has been failed.\n        n_retains (int): Number of snapshot files to retain\n            through the cleanup. Must be a positive integer for any cleanup to\n            take place. Automatic deletion of old snapshots only works when the\n            filename is string.\n        num_retain (int): Same as ``n_retains`` (deprecated).\n        autoload (bool): With this enabled, the extension automatically\n            finds the latest snapshot and loads the data to the target.\n            Automatic loading only works when the filename is a string.\n\n    Returns:\n        Snapshot extension object.\n\n    .. seealso::\n\n        - :meth:`chainer.training.extensions.snapshot`\n    "
    if 'num_retain' in kwargs:
        warnings.warn('Argument `num_retain` is deprecated. Please use `n_retains` instead', DeprecationWarning)
        kwargs['n_retains'] = kwargs.pop('num_retain')
    return snapshot(target=target, filename=filename, savefun=savefun, **kwargs)

def snapshot(savefun=None, filename='snapshot_iter_{.updater.iteration}', **kwargs):
    if False:
        i = 10
        return i + 15
    "snapshot(savefun=None, filename='snapshot_iter_{.updater.iteration}', *, target=None, condition=None, writer=None, snapshot_on_error=False, n_retains=-1, autoload=False)\n\n    Returns a trainer extension to take snapshots of the trainer.\n\n    This extension serializes the trainer object and saves it to the output\n    directory. It is used to support resuming the training loop from the saved\n    state.\n\n    This extension is called once per epoch by default. To take a\n    snapshot at a different interval, a trigger object specifying the\n    required interval can be passed along with this extension\n    to the `extend()` method of the trainer.\n\n    The default priority is -100, which is lower than that of most\n    built-in extensions.\n\n    .. note::\n       This extension first writes the serialized object to a temporary file\n       and then rename it to the target file name. Thus, if the program stops\n       right before the renaming, the temporary file might be left in the\n       output directory.\n\n    Args:\n        savefun: Function to save the trainer. It takes two arguments: the\n            output file path and the trainer object.\n            It is :meth:`chainer.serializers.save_npz` by default.\n            If ``writer`` is specified, this argument must be ``None``.\n        filename (str): Name of the file into which the trainer is serialized.\n            It can be a format string, where the trainer object is passed to\n            the :meth:`str.format` method.\n        target: Object to serialize. If it is not specified, it will\n            be the trainer object.\n        condition: Condition object. It must be a callable object that returns\n            boolean without any arguments. If it returns ``True``, the snapshot\n            will be done.\n            If not, it will be skipped. The default is a function that always\n            returns ``True``.\n        writer: Writer object.\n            It must be a callable object.\n            See below for the list of built-in writers.\n            If ``savefun`` is other than ``None``, this argument must be\n            ``None``. In that case, a\n            :class:`~chainer.training.extensions.snapshot_writers.SimpleWriter`\n            object instantiated with specified ``savefun`` argument will be\n            used.\n        snapshot_on_error (bool): Whether to take a snapshot in case trainer\n            loop has been failed.\n        n_retains (int): Number of snapshot files to retain\n            through the cleanup. Must be a positive integer for any cleanup to\n            take place. Automatic deletion of old snapshots only works when the\n            filename is string.\n        num_retain (int): Same as ``n_retains`` (deprecated).\n        autoload (bool): With this enabled, the extension\n            automatically finds the latest snapshot and loads the data\n            to the target.  Automatic loading only works when the\n            filename is a string. It is assumed that snapshots are generated\n            by :func:`chainer.serializers.save_npz` .\n\n    Returns:\n        Snapshot extension object.\n\n    .. testcode::\n       :hide:\n\n       from chainer import training\n       class Model(chainer.Link):\n           def __call__(self, x):\n               return x\n       train_iter = chainer.iterators.SerialIterator([], 1)\n       optimizer = optimizers.SGD().setup(Model())\n       updater = training.updaters.StandardUpdater(\n           train_iter, optimizer, device=0)\n       trainer = training.Trainer(updater)\n\n    .. admonition:: Using asynchronous writers\n\n        By specifying ``writer`` argument, writing operations can be made\n        asynchronous, hiding I/O overhead of snapshots.\n\n        >>> from chainer.training import extensions\n        >>> writer = extensions.snapshot_writers.ProcessWriter()\n        >>> trainer.extend(extensions.snapshot(writer=writer), trigger=(1, 'epoch'))\n\n        To change the format, such as npz or hdf5, you can pass a saving\n        function as ``savefun`` argument of the writer.\n\n        >>> from chainer.training import extensions\n        >>> from chainer import serializers\n        >>> writer = extensions.snapshot_writers.ProcessWriter(\n        ...     savefun=serializers.save_npz)\n        >>> trainer.extend(extensions.snapshot(writer=writer), trigger=(1, 'epoch'))\n\n    This is the list of built-in snapshot writers.\n\n        - :class:`chainer.training.extensions.snapshot_writers.SimpleWriter`\n        - :class:`chainer.training.extensions.snapshot_writers.ThreadWriter`\n        - :class:`chainer.training.extensions.snapshot_writers.ProcessWriter`\n        - :class:`chainer.training.extensions.snapshot_writers.ThreadQueueWriter`\n        - :class:`chainer.training.extensions.snapshot_writers.ProcessQueueWriter`\n\n    .. seealso::\n\n        - :meth:`chainer.training.extensions.snapshot_object`\n    "
    if 'num_retain' in kwargs:
        warnings.warn('Argument `num_retain` is deprecated. Please use `n_retains` instead', DeprecationWarning)
        kwargs['n_retains'] = kwargs.pop('num_retain')
    (target, condition, writer, snapshot_on_error, n_retains, autoload) = argument.parse_kwargs(kwargs, ('target', None), ('condition', None), ('writer', None), ('snapshot_on_error', False), ('n_retains', -1), ('autoload', False))
    argument.assert_kwargs_empty(kwargs)
    if savefun is not None and writer is not None:
        raise TypeError('savefun and writer arguments cannot be specified together.')
    if writer is None:
        if savefun is None:
            savefun = npz.save_npz
        writer = snapshot_writers.SimpleWriter(savefun=savefun)
    return _Snapshot(target=target, condition=condition, writer=writer, filename=filename, snapshot_on_error=snapshot_on_error, n_retains=n_retains, autoload=autoload)

def _always_true():
    if False:
        print('Hello World!')
    return True

class _Snapshot(extension.Extension):
    """Trainer extension to take snapshots.

    This extension serializes the given object and saves it to the output
    directory.

    This extension is called once per epoch by default. To take a
    snapshot at a different interval, a trigger object specifying the
    required interval can be passed along with this extension
    to the `extend()` method of the trainer.

    The default priority is -100, which is lower than that of most
    built-in extensions.
    """
    trigger = (1, 'epoch')
    priority = -100

    def __init__(self, target=None, condition=None, writer=None, filename='snapshot_iter_{.updater.iteration}', snapshot_on_error=False, n_retains=-1, autoload=False, **kwargs):
        if False:
            return 10
        if condition is None:
            condition = _always_true
        if writer is None:
            writer = snapshot_writers.SimpleWriter()
        if 'num_retain' in kwargs:
            warnings.warn('Argument `num_retain` is deprecated. Please use `n_retains` instead', DeprecationWarning)
            n_retains = kwargs['num_retain']
        self._target = target
        self.filename = filename
        self.condition = condition
        self.writer = writer
        self._snapshot_on_error = snapshot_on_error
        self.n_retains = n_retains
        self.autoload = autoload

    def initialize(self, trainer):
        if False:
            return 10
        target = trainer if self._target is None else self._target
        outdir = trainer.out
        if self.autoload:
            filename = _find_latest_snapshot(self.filename, outdir)
            if filename is None:
                if chainer.is_debug():
                    print('No snapshot file that matches {} was found'.format(self.filename))
            else:
                snapshot_file = os.path.join(outdir, filename)
                npz.load_npz(snapshot_file, target)
                if chainer.is_debug():
                    print('Snapshot loaded from', snapshot_file)
        if hasattr(self.writer, '_add_cleanup_hook') and self.n_retains > 0 and isinstance(self.filename, str):

            def _cleanup():
                if False:
                    i = 10
                    return i + 15
                files = _find_stale_snapshots(self.filename, outdir, self.n_retains)
                for file in files:
                    os.remove(os.path.join(outdir, file))
            self.writer._add_cleanup_hook(_cleanup)

    def on_error(self, trainer, exc, tb):
        if False:
            i = 10
            return i + 15
        super(_Snapshot, self).on_error(trainer, exc, tb)
        if self._snapshot_on_error:
            self._make_snapshot(trainer)

    def __call__(self, trainer):
        if False:
            i = 10
            return i + 15
        if self.condition():
            self._make_snapshot(trainer)

    def _make_snapshot(self, trainer):
        if False:
            while True:
                i = 10
        target = trainer if self._target is None else self._target
        serialized_target = npz.serialize(target)
        filename = self.filename
        if callable(filename):
            filename = filename(trainer)
        else:
            filename = filename.format(trainer)
        outdir = trainer.out
        self.writer(filename, outdir, serialized_target)

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self.writer, 'finalize'):
            self.writer.finalize()