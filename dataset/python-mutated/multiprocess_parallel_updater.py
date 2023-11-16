import multiprocessing
import warnings
import numpy
import six
import chainer
from chainer.backends import cuda
from chainer.dataset import convert
from chainer import reporter
from chainer.training.updaters import standard_updater
try:
    from cupy.cuda import nccl
    _available = True
except Exception:
    _available = False

class _Worker(multiprocessing.Process):

    def __init__(self, proc_id, pipe, master):
        if False:
            for i in range(10):
                print('nop')
        super(_Worker, self).__init__()
        self.proc_id = proc_id
        self.pipe = pipe
        self.converter = master.converter
        self.model = master._master
        self.device = master._devices[proc_id]
        self.iterator = master._mpu_iterators[proc_id]
        self.n_devices = len(master._devices)

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        (_, comm_id) = self.pipe.recv()
        self.comm = nccl.NcclCommunicator(self.n_devices, comm_id, self.proc_id)
        self.model.to_device(self.device)
        self.reporter = reporter.Reporter()
        self.reporter.add_observer('main', self.model)
        self.reporter.add_observers('main', self.model.namedlinks(skipself=True))

    def run(self):
        if False:
            print('Hello World!')
        self.device.use()
        self.setup()
        while True:
            (job, data) = self.pipe.recv()
            if job == 'finalize':
                self.device.device.synchronize()
                break
            if job == 'update':
                self.model.cleargrads()
                batch = self.converter(self.iterator.next(), self.device)
                with self.reporter.scope({}):
                    loss = _calc_loss(self.model, batch)
                self.model.cleargrads()
                loss.backward()
                del loss
                gg = gather_grads(self.model)
                nccl_data_type = _get_nccl_data_type(gg.dtype)
                null_stream = cuda.Stream.null
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size, nccl_data_type, nccl.NCCL_SUM, 0, null_stream.ptr)
                del gg
                self.model.cleargrads()
                gp = gather_params(self.model)
                nccl_data_type = _get_nccl_data_type(gp.dtype)
                self.comm.bcast(gp.data.ptr, gp.size, nccl_data_type, 0, null_stream.ptr)
                scatter_params(self.model, gp)
                del gp

class MultiprocessParallelUpdater(standard_updater.StandardUpdater):
    """Implementation of a multiprocess parallel GPU Updater.

    This is an implementation of :class:`Updater` that uses multiple GPUs
    with multi-process data parallelism. It uses Nvidia NCCL for communication
    between multiple GPUs.

    It behaves similarly to
    :class:`~chainer.training.updaters.StandardUpdater`.
    The update routine is modified to support data-parallel
    computation on multiple GPUs in one machine.
    It is based on synchronous parallel SGD: it
    parallelizes the gradient computation over a mini-batch, and updates the
    parameters only in the main device.

    It does not transfer the values collected by :class:`Reporter` in the sub
    devices to the main device. So you can only see the reported values in
    the main device.

    Args:
        iterators: List of dataset iterator for the training dataset. The
            number of the iterators must be same to the number of GPUs you use.
        optimizer: Optimizer to update parameters. The model should be attached
            to the optimizer.
        converter: Converter function to build input arrays. Each batch
            extracted by the iterator is split equally between the devices and
            then passed with corresponding ``device`` option to this function.
            :func:`~chainer.dataset.concat_examples` is used by default.
        devices: Dictionary or list of devices to which the training data is
            sent. The master device will be the first one in the list or the
            value attached to the key ``'main'``.
        auto_new_epoch (bool): If ``True``,
            :meth:`~chainer.Optimizer.new_epoch` of the main optimizer is
            automatically called when the ``is_new_epoch`` attribute of the
            main iterator is ``True``.

    """

    def __init__(self, iterators, optimizer, converter=convert.concat_examples, devices=None, auto_new_epoch=True):
        if False:
            i = 10
            return i + 15
        if not MultiprocessParallelUpdater.available():
            raise Exception('NCCL is not enabled. MultiprocessParallelUpdater requires NCCL.\nPlease reinstall CuPy after you install NCCL.\n(see https://docs-cupy.chainer.org/en/latest/install.html)')
        try:
            cuda.cupy.cuda.driver.ctxGetCurrent()
            _cuda_initialized = True
        except cuda.cupy.cuda.driver.CUDADriverError:
            _cuda_initialized = False
        if _cuda_initialized and multiprocessing.get_start_method() not in ('spawn', 'forkserver'):
            raise RuntimeError("The CUDA context has been already initialized. MultiprocessParallelUpdater assumes the context is uninitialized. Please do not call CUDA API before MultiprocessParallelUpdater creates processes or use multiprocessing.set_start_method with 'spawn' or 'forkserver' as arguments.")
        assert len(iterators) == len(devices)
        for iterator in iterators[1:]:
            assert len(iterator.dataset) == len(iterators[0].dataset)
        optim = optimizer.__class__.__name__
        if optim in ('Adam', 'AdaGrad', 'RMSprop'):
            optimizer.eps *= len(devices)
            warnings.warn('optimizer.eps is changed to {} by MultiprocessParallelUpdater for new batch size.'.format(optimizer.eps))
        elif optim in ('RMSpropGraves', 'AdaDelta'):
            optimizer.eps *= len(devices) ** 2
            warnings.warn('optimizer.eps is changed to {} by MultiprocessParallelUpdater for new batch size.'.format(optimizer.eps))
        elif hasattr(optimizer, 'lr'):
            optimizer.lr /= len(devices)
            warnings.warn('optimizer.lr is changed to {} by MultiprocessParallelUpdater for new batch size.'.format(optimizer.lr))
        super(MultiprocessParallelUpdater, self).__init__(iterator=iterators[0], optimizer=optimizer, converter=converter, auto_new_epoch=auto_new_epoch)
        if isinstance(devices, dict):
            devices = devices.copy()
            main = devices.pop('main')
            devices = list(six.itervalues(devices))
            devices = [main] + devices
        elif isinstance(devices, (list, tuple)):
            devices = list(devices)
        else:
            raise ValueError('devices argument should be either dict, list or tuple, but {} was given.'.format(type(devices)))
        if devices is None or any((device is None for device in devices)):
            raise ValueError('GPU devices must be specified.')
        self._master = optimizer.target
        self._devices = [chainer.get_device(device) for device in devices]
        self._mpu_iterators = iterators
        self._initialized = False
        self._pipes = []
        self._workers = []
        self.comm = None

    @staticmethod
    def available():
        if False:
            i = 10
            return i + 15
        return _available

    def _send_message(self, message):
        if False:
            while True:
                i = 10
        for pipe in self._pipes:
            pipe.send(message)

    def setup_workers(self):
        if False:
            while True:
                i = 10
        if self._initialized:
            return
        self._initialized = True
        self._master.cleargrads()
        for i in six.moves.range(1, len(self._devices)):
            (pipe, worker_end) = multiprocessing.Pipe()
            worker = _Worker(i, worker_end, self)
            worker.start()
            self._workers.append(worker)
            self._pipes.append(pipe)
        with chainer.using_device(self._devices[0]):
            self._master.to_device(self._devices[0])
            if len(self._devices) > 1:
                comm_id = nccl.get_unique_id()
                self._send_message(('set comm_id', comm_id))
                self.comm = nccl.NcclCommunicator(len(self._devices), comm_id, 0)

    def update_core(self):
        if False:
            while True:
                i = 10
        self.setup_workers()
        self._send_message(('update', None))
        with chainer.using_device(self._devices[0]):
            self._master.cleargrads()
            optimizer = self.get_optimizer('main')
            iterator = self.get_iterator('main')
            batch = iterator.next()
            batch = self.converter(batch, self._devices[0])
            loss = _calc_loss(self._master, batch)
            self._master.cleargrads()
            loss.backward()
            null_stream = cuda.Stream.null
            if self.comm is not None:
                gg = gather_grads(self._master)
                nccl_data_type = _get_nccl_data_type(gg.dtype)
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size, nccl_data_type, nccl.NCCL_SUM, 0, null_stream.ptr)
                scatter_grads(self._master, gg)
                del gg
            optimizer.update()
            if self.comm is not None:
                gp = gather_params(self._master)
                nccl_data_type = _get_nccl_data_type(gp.dtype)
                self.comm.bcast(gp.data.ptr, gp.size, nccl_data_type, 0, null_stream.ptr)
            if self.auto_new_epoch and iterator.is_new_epoch:
                optimizer.new_epoch(auto=True)

    def finalize(self):
        if False:
            print('Hello World!')
        self._send_message(('finalize', None))
        for worker in self._workers:
            worker.join()
        super(MultiprocessParallelUpdater, self).finalize()

def _calc_loss(model, in_arrays):
    if False:
        return 10
    if isinstance(in_arrays, tuple):
        return model(*in_arrays)
    elif isinstance(in_arrays, dict):
        return model(**in_arrays)
    else:
        return model(in_arrays)

def size_num_grads(link):
    if False:
        i = 10
        return i + 15
    'Count total size of all gradient arrays of a given link\n\n    Args:\n        link (chainer.link.Link): Target link object.\n    '
    size = 0
    num = 0
    for param in link.params():
        if param.size == 0:
            continue
        size += param.size
        num += 1
    return (size, num)

def _memcpy_gather():
    if False:
        while True:
            i = 10
    return cuda.elementwise('raw T ptrs, raw X dtypes, raw Y info', 'raw float32 dst', '\n            int id_min = id_pre;\n            int id_max = num_src;\n            while (id_max - id_min > 1) {\n                int id = (id_max + id_min) / 2;\n                if (i < info[id]) id_max = id;\n                else              id_min = id;\n            }\n            int id = id_min;\n\n            int i_dst = i;\n            int i_src = i;\n            if (id > 0) i_src -= info[id];\n\n            dst[i_dst] = 0;\n            if (ptrs[id] != NULL) {\n                if (dtypes[id] == 0) { // fp32\n                    float *src = reinterpret_cast<float *>(ptrs[id]);\n                    dst[i_dst] = src[i_src];\n                }\n                else { // fp16\n                    float16 *src = reinterpret_cast<float16 *>(ptrs[id]);\n                    dst[i_dst] = static_cast<float>(src[i_src]);\n                }\n            }\n            id_pre = id;\n        ', '_memcpy_gather', loop_prep='\n                int num_src = info[0];\n                int id_pre = 0;\n            ')

def _gather(link, target):
    if False:
        for i in range(10):
            print('nop')
    (size, num) = size_num_grads(link)
    ptrs = numpy.empty(num, dtype=numpy.uint64)
    dtypes = numpy.empty(num, dtype=numpy.int8)
    info = numpy.empty(num + 1, dtype=numpy.int32)
    info[0] = 0
    i = 0
    for (_, param) in sorted(link.namedparams()):
        if param.size == 0:
            continue
        ptrs[i] = 0
        d = getattr(param, target)
        if d is not None:
            ptrs[i] = d.data.ptr
        dtypes[i] = 0
        if param.dtype == numpy.float16:
            dtypes[i] = 1
        info[i + 1] = info[i] + param.size
        i += 1
    info[0] = num
    ptrs = cuda.to_gpu(ptrs)
    dtypes = cuda.to_gpu(dtypes)
    info = cuda.to_gpu(info)
    return _memcpy_gather()(ptrs, dtypes, info, size=size)

def gather_grads(link):
    if False:
        for i in range(10):
            print('nop')
    'Put together all gradient arrays and make a single array\n\n    Args:\n        link (chainer.link.Link): Target link object.\n    Return:\n        cupy.ndarray\n    '
    if link.xp is numpy:
        raise RuntimeError('gather_grads works only on GPU.')
    return _gather(link, 'grad')

def gather_params(link):
    if False:
        for i in range(10):
            print('nop')
    'Put together all gradient arrays and make a single array\n\n    Args:\n        link (chainer.link.Link): Target link object.\n    Return:\n        cupy.ndarray\n    '
    if link.xp is numpy:
        raise RuntimeError('Link.gather_params works only on GPU.')
    return _gather(link, 'data')

def _memcpy_scatter():
    if False:
        for i in range(10):
            print('nop')
    return cuda.elementwise('raw T ptrs, raw X dtypes, raw Y info, raw float32 array', '', '\n            int id_min = id_pre;\n            int id_max = num_src;\n            while (id_max - id_min > 1) {\n                int id = (id_max + id_min) / 2;\n                if (i < info[id]) id_max = id;\n                else              id_min = id;\n            }\n            int id = id_min;\n\n            int i_src = i;\n            int i_dst = i;\n            if (id > 0) i_dst -= info[id];\n\n            if (ptrs[id] != NULL) {\n                if (dtypes[id] == 0) { // fp32\n                    float *dst = reinterpret_cast<float *>(ptrs[id]);\n                    dst[i_dst] = array[i_src];\n                }\n                else { // fp16\n                    float16 *dst = reinterpret_cast<float16 *>(ptrs[id]);\n                    dst[i_dst] = static_cast<float16>(array[i_src]);\n                }\n            }\n            id_pre = id;\n        ', '_memcpy_scatter', loop_prep='\n                int num_src = info[0];\n                int id_pre = 0;\n            ')

def _scatter(link, array, target):
    if False:
        return 10
    (size, num) = size_num_grads(link)
    ptrs = numpy.zeros(num, dtype=numpy.uint64)
    dtypes = numpy.zeros(num, dtype=numpy.int8)
    info = numpy.zeros(num + 1, dtype=numpy.int32)
    info[0] = 0
    i = 0
    for (_, param) in sorted(link.namedparams()):
        if param.size == 0:
            continue
        ptrs[i] = 0
        d = getattr(param, target)
        if d is None:
            d = cuda.cupy.zeros(param.shape, dtype=param.dtype)
            setattr(param, target, d)
        ptrs[i] = d.data.ptr
        dtypes[i] = 0
        if param.dtype == numpy.float16:
            dtypes[i] = 1
        info[i + 1] = info[i] + param.size
        i += 1
    if i != num:
        raise ()
    info[0] = num
    ptrs = cuda.to_gpu(ptrs)
    dtypes = cuda.to_gpu(dtypes)
    info = cuda.to_gpu(info)
    return _memcpy_scatter()(ptrs, dtypes, info, array, size=size)

def scatter_grads(link, array):
    if False:
        return 10
    'Put back contents of the specified array to the related gradient arrays\n\n    Args:\n        link (chainer.link.Link): Target link object.\n        array (cupy.ndarray): gathered array created by gather_grads()\n    '
    return _scatter(link, array, 'grad')

def scatter_params(link, array):
    if False:
        while True:
            i = 10
    'Put back contents of the specified array to the related gradient arrays\n\n    Args:\n        link (chainer.link.Link): Target link object.\n        array (cupy.ndarray): gathered array created by gather_params()\n    '
    return _scatter(link, array, 'data')

def _get_nccl_data_type(dtype):
    if False:
        i = 10
        return i + 15
    'Get data type for NCCL'
    if dtype == numpy.float32:
        nccl_data_type = nccl.NCCL_FLOAT
    elif dtype == numpy.float16:
        nccl_data_type = nccl.NCCL_HALF
    elif dtype == numpy.float64:
        nccl_data_type = nccl.NCCL_DOUBLE
    else:
        raise RuntimeError('Unexpected data type:{}'.format(dtype))
    return nccl_data_type