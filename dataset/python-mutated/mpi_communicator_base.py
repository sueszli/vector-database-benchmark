import mpi4py
import numpy
import chainer
import chainer.backends
import chainer.utils
from chainer.utils import collections_abc
from chainermn.communicators import _communication_utility
from chainermn.communicators._communication_utility import chunked_bcast_obj
from chainermn.communicators import _memory_utility
from chainermn.communicators import communicator_base
import chainerx
_dtype_mpi_type = {numpy.dtype(numpy.int32): mpi4py.MPI._typedict['i'], numpy.dtype(numpy.int64): mpi4py.MPI._typedict['l'], numpy.dtype(numpy.float16): mpi4py.MPI._typedict['f'], numpy.dtype(numpy.float32): mpi4py.MPI._typedict['f'], numpy.dtype(numpy.float64): mpi4py.MPI._typedict['d']}

def _check_dtype(caller, msgtype):
    if False:
        while True:
            i = 10
    dtype = msgtype.dtype
    if dtype not in _dtype_mpi_type.keys():
        raise TypeError('{} does not support dtype {}'.format(caller, dtype))

def _check_dtypes_are_same(msgtypes):
    if False:
        print('Hello World!')
    dtypes = [msgtype.dtype for msgtype in msgtypes]
    if any((dtypes[0] != dtype for dtype in dtypes)):
        raise TypeError('all dtypes must be the same')

def _is_numpy_array(array):
    if False:
        while True:
            i = 10
    return isinstance(array, numpy.ndarray)

def _is_cupy_array(array):
    if False:
        return 10
    return chainer.backend.get_array_module(array) is not numpy

def _cnt_to_dsp(cnt):
    if False:
        while True:
            i = 10
    'Utility to convert length array to cumulative array.'
    return [0] + numpy.cumsum(cnt)[:-1].tolist()

def _get_mpi_type(msgtype):
    if False:
        while True:
            i = 10
    dtype = msgtype.dtype
    if dtype not in _dtype_mpi_type.keys():
        raise TypeError('dtype {} is not supported by MpiCommunicator'.format(dtype))
    return _dtype_mpi_type[dtype]

class _MessageType(object):

    def __init__(self, obj):
        if False:
            while True:
                i = 10
        if _is_numpy_array(obj) or _is_cupy_array(obj):
            self.is_host = _is_numpy_array(obj)
            self.is_tuple = False
            self.narr = 1
            self.ndims = [obj.ndim]
            self.shapes = [obj.shape]
            self.dtype = obj.dtype
        elif isinstance(obj, collections_abc.Iterable):
            if all(map(_is_numpy_array, obj)):
                self.is_host = True
            elif all(map(_is_cupy_array, obj)):
                self.is_host = False
            else:
                raise ValueError('All message objects must be either numpy or cupy arrays.')
            self.is_tuple = True
            self.narr = len(obj)
            self.ndims = [x.ndim for x in obj]
            self.shapes = [x.shape for x in obj]
            dtypes = [x.dtype for x in obj]
            if not all((dtype == dtypes[0] for dtype in dtypes)):
                raise TypeError('Message objects must be the same dtype')
            self.dtype = dtypes[0]
        else:
            raise TypeError('Message object must be numpy/cupy array or its tuple.')

    def get_array_module(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_host:
            return numpy
        else:
            import cupy
            return cupy

class MpiCommunicatorBase(communicator_base.CommunicatorBase):
    """MpiCommunicatorBase

    Implementation of communicator interface defined by
    :class:`CommunicatorBase`. This communicator assumes MPI4py and
    all ChainerMN processes are invoked by ``mpirun`` (``mpiexec``)
    command. Although this lacks several important methods such as
    ``multi_node_mean_grad`` to be impelmented with speficic algorithm. See
    hierarchical communicator or pure_nccl communicator for example.

    """

    def __init__(self, mpi_comm):
        if False:
            return 10
        self.mpi_comm = mpi_comm
        self._init_ranks()
        with self.config_scope():
            self.batched_copy = False

    @property
    def rank(self):
        if False:
            for i in range(10):
                print('nop')
        return self.mpi_comm.rank

    @property
    def size(self):
        if False:
            return 10
        return self.mpi_comm.size

    @property
    def intra_rank(self):
        if False:
            i = 10
            return i + 15
        return self._intra_rank

    @property
    def intra_size(self):
        if False:
            for i in range(10):
                print('nop')
        return self._intra_size

    @property
    def inter_rank(self):
        if False:
            for i in range(10):
                print('nop')
        return self._inter_rank

    @property
    def inter_size(self):
        if False:
            for i in range(10):
                print('nop')
        return self._inter_size

    def set_config(self, name, value=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if name == 'batched_copy':
            with self.config_scope():
                self.batched_copy = value
        else:
            return super(MpiCommunicatorBase, self).set_config(name, **kwargs)

    def get_config(self, name=None):
        if False:
            while True:
                i = 10
        if name == 'batched_copy':
            return self.batched_copy
        else:
            return super(MpiCommunicatorBase, self).get_config(name)

    def split(self, color, key):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__(mpi_comm=self.mpi_comm.Split(color, key))

    def alltoall(self, xs):
        if False:
            while True:
                i = 10
        'A primitive of inter-process all-to-all function.\n\n        This method tries to invoke all-to-all communication within the\n        communicator. All processes in the communicator are expected to\n        invoke ``alltoall()``. This method relies on mpi4py fast communication\n        optimized for numpy arrays, as well as ``send()`` and ``recv()``.\n\n        If ``xs`` is numpy array, the returned array will also be allocated\n        as numpy array. Additionally, when ``xs`` is cupy array, the returned\n        array will be placed at current device\n        (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)\n        regardless of which device the argument is placed at remote nodes.\n\n        Args:\n            xs (tuple of numpy/cupy array)\n\n        Returns:\n            ys (tuple of numpy/cupy array):\n                Received arrays. The length of tuple equals to\n                the communicator size.\n        '
        chainer.utils.experimental('chainermn.communicators.MpiCommunicatorBase.alltoall')
        if len(xs) != self.size:
            raise ValueError('The length of data must be same as communicator size.')
        msgtypes = [_MessageType(x) for x in xs]
        for msgtype in msgtypes:
            _check_dtype('alltoall', msgtype)
        _check_dtypes_are_same(msgtypes)
        send_msgtype = msgtypes[0]
        msgtypes = self.mpi_comm.alltoall(msgtypes)
        _check_dtypes_are_same(msgtypes)
        recv_msgtype = msgtypes[0]
        slens = [x.size for x in xs]
        xp = chainer.backend.get_array_module(*xs)
        sbuf = xp.hstack([x.reshape(-1) for x in xs])
        shapes = [msgtype.shapes[0] for msgtype in msgtypes]
        rlens = [chainer.utils.size_of_shape(s) for s in shapes]
        rbuf = xp.empty([sum(rlens)], dtype=msgtype.dtype)
        if xp is not numpy:
            sbuf = _memory_utility.get_device_memory_pointer(sbuf)
            chainer.cuda.Stream.null.synchronize()
        self.mpi_comm.Alltoallv([sbuf, (slens, _cnt_to_dsp(slens)), _get_mpi_type(send_msgtype)], [_memory_utility.get_device_memory_pointer(rbuf), (rlens, _cnt_to_dsp(rlens)), _get_mpi_type(recv_msgtype)])
        ys = [rbuf[i:i + l].reshape(s) for (i, l, s) in zip(_cnt_to_dsp(rlens), rlens, shapes)]
        return tuple(ys)

    def send(self, data, dest, tag):
        if False:
            while True:
                i = 10
        'A primitive for inter-process transmitter.\n\n        This method sends numpy-array to target process.\n        The target process is expected to invoke ``recv()``.\n        This method relies on mpi4py fast communication optimized for\n        numpy arrays, which discards any information attached to\n        chainer.Variable objects. Please be sure.\n\n        Args:\n            data: data to be sent (tuple, list or raw numpy/cupy array)\n            dest (int): Target process specifier.\n            tag (int): Message ID (MPI feature).\n\n        '
        chainer.utils.experimental('chainermn.communicators.MpiCommunicatorBase.send')
        msgtype = _MessageType(data)
        _check_dtype('send', msgtype)
        "We use ssend() instead of send() to pass unittests.\n        If we don't use it, an error occurs in\n        test_point_to_point_communication.py\n        when using MVAPICH2-2.2 and GPUs.\n        "
        self.mpi_comm.ssend(msgtype, dest=dest, tag=tag)
        if not msgtype.is_tuple:
            data = [data]
        for array in data:
            if numpy.float16 == array.dtype:
                array = array.astype(numpy.float32)
            if chainer.backend.get_array_module(array) is not numpy:
                chainer.cuda.Stream.null.synchronize()
                array = (_memory_utility.get_device_memory_pointer(array), _get_mpi_type(msgtype))
            else:
                array = numpy.ascontiguousarray(array)
            'We use Ssend() for the same reason as using ssend().'
            self.mpi_comm.Ssend(array, dest=dest, tag=tag)

    def recv(self, source, tag):
        if False:
            while True:
                i = 10
        'A primitive of inter-process receiver.\n\n        This method tries to receive numpy-array from target process.\n        The target process is expected to invoke ``send()``.\n        This method relies on mpi4py fast communication optimized for\n        numpy arrays, which discards any information attached to\n        chainer.Variable objects. Please be sure.\n\n        If the corresponding ``send()`` is invoked with cupy array,\n        the returned array will be placed at current device\n        (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)\n        regardless of which device the argument is placed at remote nodes.\n\n        Args:\n            source (int): Target process specifier.\n            tag (int): Message ID (MPI feature).\n\n        Returns:\n            data (tuple of numpy/cupy array or numpy/cupy array):\n                Received data. If ``send()`` is invoked with tuple data,\n                it is also tuple. Otherwise, it is a vanilla numpy/cupy array.\n        '
        chainer.utils.experimental('chainermn.communicators.MpiCommunicatorBase.recv')
        msgtype = self.mpi_comm.recv(source=source, tag=tag)
        xp = msgtype.get_array_module()
        if numpy.float16 == msgtype.dtype:
            comm_dtype = numpy.float32
        else:
            comm_dtype = msgtype.dtype
        if msgtype.is_tuple:
            msg = []
            for shape in msgtype.shapes:
                buf = xp.empty([chainer.utils.size_of_shape(shape)], dtype=comm_dtype)
                rtype = _get_mpi_type(msgtype)
                self.mpi_comm.Recv(_memory_utility.array_to_buffer_object(buf, rtype), source=source, tag=tag)
                if numpy.float16 == msgtype.dtype:
                    buf = buf.astype(numpy.float16)
                msg.append(buf.reshape(shape))
            return tuple(msg)
        else:
            assert len(msgtype.shapes) == 1
            shape = msgtype.shapes[0]
            buf = xp.empty([chainer.utils.size_of_shape(shape)], dtype=comm_dtype)
            rtype = _get_mpi_type(msgtype)
            self.mpi_comm.Recv(_memory_utility.array_to_buffer_object(buf, rtype), source=source, tag=tag)
            if numpy.float16 == msgtype.dtype:
                buf = buf.astype(numpy.float16)
            return buf.reshape(shape)

    def bcast(self, x, root=0):
        if False:
            i = 10
            return i + 15
        'A primitive of inter-process broadcast communication.\n\n        This method tries to invoke broadcast communication within the\n        communicator. All processes in the communicator are expected to\n        invoke ``broadcast()``. This method relies on mpi4py fast communication\n        optimized for numpy arrays, as well as ``send()`` and ``recv()``.\n\n        If ``bcast()`` is invoked with cupy array in the root process,\n        the returned array will be placed at current device\n        (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)\n        regardless of which device the argument is placed at remote nodes.\n\n        Args:\n            x (numpy/cupy array): Array to be broadcasted.\n            root (int): Rank of root process.\n\n        Returns:\n            ys (tuple of numpy/cupy array): Received arrays.\n        '
        chainer.utils.experimental('chainermn.communicators.MpiCommunicatorBase.bcast')
        is_master = self.mpi_comm.rank == root
        if is_master:
            msgtype = _MessageType(x)
            _check_dtype('bcast', msgtype)
            if msgtype.is_tuple:
                raise TypeError('Tuple data cannot be broadcasted')
            msgtype = self.mpi_comm.bcast(msgtype, root)
            shape = msgtype.shapes[0]
            buf = _memory_utility.array_to_buffer_object(x, _get_mpi_type(msgtype))
            self.mpi_comm.Bcast(buf, root)
            return x
        else:
            msgtype = self.mpi_comm.bcast(None, root)
            xp = msgtype.get_array_module()
            shape = msgtype.shapes[0]
            buf = xp.empty([chainer.utils.size_of_shape(shape)], dtype=msgtype.dtype)
            buftype = _get_mpi_type(msgtype)
            self.mpi_comm.Bcast(_memory_utility.array_to_buffer_object(buf, buftype), root)
            return buf.reshape(shape)

    def gather(self, x, root=0):
        if False:
            print('Hello World!')
        'A primitive of inter-process gather communication.\n\n        This method tries to invoke gather communication within the\n        communicator. All processes in the communicator are expected to\n        invoke ``gather()``. This method relies on mpi4py fast communication\n        optimized for numpy arrays, as well as ``send()`` and ``recv()``.\n\n        If ``x`` is numpy array, the received data will also be allocated\n        as numpy array. Additionally, when ``x`` is cupy array, the returned\n        array will be placed at current device\n        (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)\n        regardless of which device the argument is placed at remote nodes.\n\n        Args:\n            x (numpy/cupy array): Array to be gathered.\n            root (int): Rank of root process.\n\n        Returns:\n            ys (tuple of numpy/cupy array):\n                Received arrays. ``None`` for non-root processes.\n        '
        chainer.utils.experimental('chainermn.communicators.MpiCommunicatorBase.gather')
        is_master = self.mpi_comm.rank == root
        msgtype = _MessageType(x)
        _check_dtype('gather', msgtype)
        msgtypes = self.mpi_comm.gather(msgtype, root)
        if is_master:
            _check_dtypes_are_same(msgtypes)
            for msgtype in msgtypes:
                if msgtype.is_tuple:
                    raise TypeError('gather cannot handle tuple data')
                assert len(msgtype.shapes) == 1
            xp = chainer.backend.get_array_module(x)
            sbuf = _memory_utility.array_to_buffer_object(x, _get_mpi_type(msgtype))
            shapes = [mty.shapes[0] for mty in msgtypes]
            rlens = [chainer.utils.size_of_shape(s) for s in shapes]
            rbuf = xp.empty([sum(rlens)], dtype=msgtype.dtype)
            if xp is not numpy:
                chainer.cuda.Stream.null.synchronize()
            self.mpi_comm.Gatherv(sbuf, [_memory_utility.get_device_memory_pointer(rbuf), (rlens, _cnt_to_dsp(rlens)), _get_mpi_type(msgtype)], root)
            ys = [rbuf[i:i + l].reshape(s) for (i, l, s) in zip(_cnt_to_dsp(rlens), rlens, shapes)]
            return tuple(ys)
        else:
            sbuf = _memory_utility.array_to_buffer_object(x, _get_mpi_type(msgtype))
            self.mpi_comm.Gatherv(sbuf, None, root)
            return None

    def allgather(self, x):
        if False:
            i = 10
            return i + 15
        chainer.utils.experimental('chainermn.communicators.MpiCommunicatorBase.allgather')
        msgtype = _MessageType(x)
        _check_dtype('allgather', msgtype)
        msgtypes = self.mpi_comm.allgather(msgtype)
        _check_dtypes_are_same(msgtypes)
        for msgtype in msgtypes:
            if msgtype.is_tuple:
                raise TypeError('allgather cannot handle tuple data')
            assert len(msgtype.shapes) == 1
        xp = chainer.backend.get_array_module(x)
        shapes = [msgtype.shapes[0] for msgtype in msgtypes]
        sbuf = _memory_utility.array_to_buffer_object(x, _get_mpi_type(msgtype))
        rlens = [chainer.utils.size_of_shape(s) for s in shapes]
        rbuf = xp.empty([sum(rlens)], dtype=msgtype.dtype)
        if xp is not numpy:
            chainer.cuda.Stream.null.synchronize()
        self.mpi_comm.Allgatherv(sbuf, [_memory_utility.get_device_memory_pointer(rbuf), (rlens, _cnt_to_dsp(rlens)), _get_mpi_type(msgtype)])
        ys = [rbuf[i:i + l].reshape(s) for (i, l, s) in zip(_cnt_to_dsp(rlens), rlens, shapes)]
        return tuple(ys)

    def allreduce(self, x):
        if False:
            while True:
                i = 10
        'A primitive of inter-process allreduce communication.\n\n        This method tries to invoke allreduce communication within the\n        communicator. All processes in the communicator are expected to\n        invoke ``allreduce()``. This method relies on mpi4py fast communication\n        optimized for numpy arrays, as well as ``send()`` and ``recv()``.\n\n        Note that this method can only handle the same shapes of data\n        over all processes, and cannot handle tuple data.\n\n        If ``x`` is numpy array, the received data will also be allocated\n        as numpy array. Additionally, when ``x`` is cupy array, the returned\n        array will be placed at current device\n        (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)\n        regardless of which device the argument is placed at remote nodes.\n\n        Args:\n            x (numpy/cupy array): An array to apply allreduce operation.\n\n        Returns:\n            ys (numpy/cupy array): An array that allreduce (currently SUM only)\n                has been applied.\n\n        '
        msgtype = _MessageType(x)
        _check_dtype('allreduce', msgtype)
        if msgtype.is_tuple:
            raise TypeError('allreduce cannot handle tuple data')
        xp = chainer.backend.get_array_module(x)
        sbuf = _memory_utility.array_to_buffer_object(x, _get_mpi_type(msgtype))
        shape = msgtype.shapes[0]
        dbuf = xp.empty([chainer.utils.size_of_shape(shape)], dtype=msgtype.dtype)
        dbuf_buffer_obj = _memory_utility.array_to_buffer_object(dbuf, _get_mpi_type(msgtype))
        self.mpi_comm.Allreduce(sbuf, dbuf_buffer_obj)
        return dbuf.reshape(shape)

    def scatter(self, xs, root=0):
        if False:
            return 10
        'A primitive of inter-process scatter communication.\n\n        This method tries to invoke scatter communication within the\n        communicator. All processes in the communicator are expected to\n        invoke ``scatter()``. This method relies on mpi4py fast communication\n        optimized for numpy arrays, as well as ``send()`` and ``recv()``.\n\n        If ``xs`` is tuple, each element is send to different processes.\n        The length of the tuple must be the same as the communicator size.\n        If ``xs`` is ``numpy.ndarrray``, it is splitted with the first\n        axis and sent to different processes. For slave processes, ``xs``\n        is allowed to be any value (will be ignored).\n\n        If ``scatter()`` is invoked with cupy array in the root process,\n        the returned array will be placed at current device\n        (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)\n        regardless of which device the argument is placed at remote nodes.\n\n        Args:\n            xs (tuple of numpy/cupy array): Arrays to be scattered.\n            root (int): Rank of root process.\n\n        Returns:\n            ys (numpy/cupy array): Received arrays.\n        '
        chainer.utils.experimental('chainermn.communicators.CommunicatorBase.scatter')
        is_master = self.mpi_comm.rank == root
        if is_master:
            msgtype = _MessageType(xs)
            _check_dtype('scatter', msgtype)
            if msgtype.is_tuple:
                if len(msgtype.shapes) != self.size:
                    raise ValueError('the length of xs must be consistent with communicator size')
                xp = chainer.backend.get_array_module(*xs)
                msgtype = tuple([_MessageType(x) for x in xs])
                shapes = [mty.shapes[0] for mty in msgtype]
                xs = xp.concatenate([x.reshape(1, -1) for x in xs], axis=1)
            else:
                assert len(msgtype.shapes) == 1
                if msgtype.shapes[0][0] != self.mpi_comm.size:
                    raise ValueError('scatter received inconsistent number of inputs with communicator size')
                xp = chainer.backend.get_array_module(xs)
                msgtype = tuple([_MessageType(xs[0]) for _ in range(self.size)])
                shapes = [xs.shape[1:] for _ in range(self.size)]
            msgtype = self.mpi_comm.scatter(msgtype, root)
            shape = msgtype.shapes[0]
            slens = [chainer.utils.size_of_shape(s) for s in shapes]
            sbuf = _memory_utility.get_device_memory_pointer(xs)
            rbuf = xp.empty([chainer.utils.size_of_shape(shape)], dtype=msgtype.dtype)
            rtype = _get_mpi_type(msgtype)
            if xp is not numpy:
                chainer.cuda.Stream.null.synchronize()
            self.mpi_comm.Scatterv([sbuf, (slens, _cnt_to_dsp(slens)), _get_mpi_type(msgtype)], _memory_utility.array_to_buffer_object(rbuf, rtype), root)
            return rbuf.reshape(shape)
        else:
            msgtypes = self.mpi_comm.scatter(None, root)
            xp = msgtypes.get_array_module()
            shape = msgtypes.shapes[0]
            rbuf = xp.empty([chainer.utils.size_of_shape(shape)], dtype=msgtypes.dtype)
            rtype = _get_mpi_type(msgtypes)
            self.mpi_comm.Scatterv(None, _memory_utility.array_to_buffer_object(rbuf, rtype), root)
            return rbuf.reshape(shape)

    def _check_obj_type_for_chainerx(self, obj):
        if False:
            return 10
        if None is obj:
            return False
        elif type(obj) in [list, tuple, set]:
            for item in obj:
                xp = chainer.backend.get_array_module(item)
                if xp == chainerx and item.device.name.startswith('cuda'):
                    return True
        elif type(obj) is dict:
            for (key, value) in obj.items():
                xp = chainer.backend.get_array_module(key)
                if xp == chainerx and key.device.name.startswith('cuda'):
                    return True
                xp = chainer.backend.get_array_module(value)
                if xp == chainerx and value.device.name.startswith('cuda'):
                    return True
        else:
            xp = chainer.backend.get_array_module(obj)
            if xp == chainerx and obj.device.name.startswith('cuda'):
                return True
        return False

    def send_obj(self, obj, dest, tag=0):
        if False:
            print('Hello World!')
        if self._check_obj_type_for_chainerx(obj):
            raise ValueError('calling send_obj on chainerx                 with cuda is not supported')
        self.mpi_comm.send(obj, dest=dest, tag=tag)

    def recv_obj(self, source, status=None, tag=mpi4py.MPI.ANY_TAG):
        if False:
            print('Hello World!')
        return self.mpi_comm.recv(source=source, status=status, tag=tag)

    def bcast_obj(self, obj, max_buf_len=256 * 1024 * 1024, root=0):
        if False:
            print('Hello World!')
        if self._check_obj_type_for_chainerx(obj):
            raise ValueError('calling bcast_obj on chainerx                 with cuda is not supported')
        return chunked_bcast_obj(obj, self.mpi_comm, max_buf_len=max_buf_len, root=root)

    def gather_obj(self, obj, root=0):
        if False:
            i = 10
            return i + 15
        if self._check_obj_type_for_chainerx(obj):
            raise ValueError('calling gather_obj on chainerx                 with cuda is not supported')
        return self.mpi_comm.gather(obj, root=root)

    def allreduce_obj(self, obj):
        if False:
            return 10
        if self._check_obj_type_for_chainerx(obj):
            raise ValueError('calling allreduce_obj on chainerx                 with cuda is not supported')
        return self.mpi_comm.allreduce(obj)

    def bcast_data(self, model):
        if False:
            while True:
                i = 10
        for (_, param) in sorted(model.namedparams()):
            if param.data is not None:
                data = param.data
                is_float16 = param.data.dtype == numpy.float16
                if is_float16:
                    data = data.astype(numpy.float32)
                buf = _memory_utility.array_to_buffer_object(data)
                self.mpi_comm.Bcast(buf)
                if is_float16:
                    param.array[...] = data.astype(numpy.float16)

    def _init_ranks(self):
        if False:
            print('Hello World!')
        my_ranks = _communication_utility.init_ranks(self.mpi_comm)
        assert my_ranks[0] == self.mpi_comm.rank
        self._intra_rank = my_ranks[1]
        self._intra_size = my_ranks[2]
        self._inter_rank = my_ranks[3]
        self._inter_size = my_ranks[4]

    def _check_ready_to_allreduce(self, array_a, array_b):
        if False:
            while True:
                i = 10
        my_shapes = ((None if array_a is None else array_a.shape, None if array_a is None else array_a.dtype), array_b.shape, array_b.dtype)
        all_shapes = self.gather_obj((self.rank, my_shapes))
        if self.rank == 0:
            for (rank, shapes) in all_shapes:
                if my_shapes != shapes:
                    raise ValueError('Shape does not match: {} at rank 0 while {} at rank {}'.format(my_shapes, shapes, rank))

    def _ensure_all_finite(self, array):
        if False:
            print('Hello World!')
        xp = chainer.backend.get_array_module(array)
        if not xp.isfinite(array).all():
            raise ValueError('Parameters diverged after allreduce.')

    def _multi_node_mean(self, sendbuf, recvbuf):
        if False:
            i = 10
            return i + 15
        'Compute mean of each element on each processes.\n\n        The function compute mean of each element in ``sendbuf`` on each\n        processes. The result is stored in ``recvbuf``.\n\n        If ``sendbuf`` is ``None``, the function compute mean of each element\n        in ``recvbuf`` on each processes and replaces ``recvbuf` with the\n        computed mean.\n\n        Args:\n            sendbuf (numpy/cupy array): Input arrays.\n            recvbuf (numpy/cupy array): Output arrays.\n\n        '
        if chainer.is_debug():
            self._check_ready_to_allreduce(sendbuf, recvbuf)
        is_float16 = recvbuf.dtype == numpy.float16
        if sendbuf is None:
            buffer_a = mpi4py.MPI.IN_PLACE
        elif is_float16:
            assert sendbuf.dtype == recvbuf.dtype
            buffer_a = _memory_utility.array_to_buffer_object(sendbuf.astype(numpy.float32))
        else:
            buffer_a = _memory_utility.array_to_buffer_object(sendbuf)
        if is_float16:
            array_b32 = recvbuf.astype(numpy.float32)
        else:
            array_b32 = recvbuf
        buffer_b = _memory_utility.array_to_buffer_object(array_b32)
        self.mpi_comm.Allreduce(buffer_a, buffer_b)
        if is_float16:
            recvbuf[...] = array_b32.astype(numpy.float16)
        recvbuf *= 1.0 / self.mpi_comm.size
        if chainer.is_debug():
            self._ensure_all_finite(recvbuf)

    def _pack_params_to_buffer(self, params, attr_name, buffer, allreduce_grad_dtype, zero_fill, stream=None):
        if False:
            while True:
                i = 10
        if self.batched_copy:
            params_data = _memory_utility.ParamsData(params, attr_name, zero_fill)
            _memory_utility._batched_pack_params(params_data, buffer, allreduce_grad_dtype, stream=stream)
            self.params_data = params_data
        else:
            _memory_utility.pack_params(params, attr_name, buffer, transfer_dtype=allreduce_grad_dtype, zero_fill=zero_fill, stream=stream)

    def _unpack_params_from_buffer(self, params, attr_name, buffer, allreduce_grad_dtype, zero_fill, stream=None):
        if False:
            i = 10
            return i + 15
        if self.batched_copy:
            if self.params_data is not None:
                params_data = self.params_data
                self.params_data = None
            else:
                params_data = _memory_utility.ParamsData(params, attr_name, zero_fill)
            _memory_utility._batched_unpack_params(params_data, buffer, allreduce_grad_dtype, stream=stream)
            return
        else:
            _memory_utility.unpack_params(params, attr_name, buffer, allreduce_grad_dtype, zero_fill, stream)