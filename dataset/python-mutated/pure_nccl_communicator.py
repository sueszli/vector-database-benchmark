import warnings
import chainer.cuda
from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn.communicators import mpi_communicator_base
from chainermn import nccl
import numpy as np

class PureNcclCommunicator(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm):
        if False:
            print('Hello World!')
        super(PureNcclCommunicator, self).__init__(mpi_comm)
        if not nccl._available:
            raise RuntimeError('PureNcclCommunicator requires NCCL 2.0+, but NCCL is not available.')
        if nccl.get_build_version() < 2000:
            raise RuntimeError('PureNcclCommunicator requires NCCL 2.0+, but found {}.'.format(nccl.get_build_version()))
        if nccl.get_version() < 2302:
            warnings.warn('NCCL 2.2 and older versions are deprecated.', DeprecationWarning)
        self.nccl_comm = None
        self.gpu_tmp_buffer = _memory_utility.DeviceMemory()
        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()
        with self.config_scope():
            self.allreduce_grad_dtype = None
        self.grad_dtype_to_allreduce_dtype_kernel = None
        self.allreduce_dtype_to_grad_dtype_kernel = None
        self.params_data = None

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        super(PureNcclCommunicator, self).finalize()
        if self.nccl_comm is not None:
            chainer.cuda.Stream.null.synchronize()
            self.mpi_comm.barrier()
            self.nccl_comm.destroy()
            self.nccl_comm = None

    def _init_comms(self):
        if False:
            return 10
        if self.nccl_comm is not None:
            return
        self.nccl_comm = _communication_utility.init_nccl_comm(self.mpi_comm)

    def set_config(self, name, value=True, **kwargs):
        if False:
            i = 10
            return i + 15
        if name == 'allreduce_grad_dtype':
            if value is not None:
                allreduce_grad_dtype = np.dtype(value)
                if allreduce_grad_dtype.kind != 'f':
                    raise ValueError('allreduce_grad_dtype must benumpy.float16, numpy.float32,numpy.float64, or None.')
            else:
                allreduce_grad_dtype = None
            with self.config_scope():
                self.allreduce_grad_dtype = allreduce_grad_dtype
        else:
            super(PureNcclCommunicator, self).set_config(name, **kwargs)

    def get_config(self, name=None):
        if False:
            print('Hello World!')
        if name == 'allreduce_grad_dtype':
            return self.allreduce_grad_dtype
        else:
            return super(PureNcclCommunicator, self).get_config(name)

    def bcast_data(self, model):
        if False:
            while True:
                i = 10
        self._init_comms()
        params = _memory_utility.extract_params_set_data(model)
        data_dtype = chainer.get_dtype()
        n_elems = sum((param.data.size for param in params))
        data_grad_n_bytes = data_dtype.itemsize * n_elems
        if self.gpu_tmp_buffer.size != data_grad_n_bytes:
            self.gpu_tmp_buffer.assign(data_grad_n_bytes)
        stream = chainer.cuda.Stream.null
        _memory_utility.pack_params(params, 'data', self.gpu_tmp_buffer, data_dtype, False, stream)
        self.nccl_comm.bcast(self.gpu_tmp_buffer.ptr(), n_elems, _communication_utility._get_nccl_type_id(data_dtype), 0, stream.ptr)
        _memory_utility.unpack_params(params, 'data', self.gpu_tmp_buffer, data_dtype, False, stream)

    def multi_node_mean_grad(self, model, zero_fill=False):
        if False:
            return 10
        stream = chainer.cuda.Stream.null
        self._multi_node_mean_grad_async(model, zero_fill, stream)

    def _multi_node_mean_grad_async(self, model, zero_fill, stream):
        if False:
            return 10
        self._init_comms()
        params = _memory_utility.extract_params_set_grad(model, zero_fill)
        if self.allreduce_grad_dtype is not None:
            allreduce_grad_dtype = self.allreduce_grad_dtype
        else:
            allreduce_grad_dtype = chainer.get_dtype()
        assert allreduce_grad_dtype is not None
        n_elems = _memory_utility.count_grad_elements(params, zero_fill)
        needs_sync = self._prepare_allreduce_pack_buffer(allreduce_grad_dtype, n_elems)
        if stream != chainer.cuda.Stream.null and needs_sync:
            chainer.cuda.Stream.null.synchronize()
        self._pack_params_to_buffer(params, 'grad', self.gpu_buffer_a, allreduce_grad_dtype, zero_fill, stream)
        self._multi_node_mean_nccl(self.gpu_buffer_a, self.gpu_buffer_b, n_elems, allreduce_grad_dtype, stream)
        self._unpack_params_from_buffer(params, 'grad', self.gpu_buffer_b, allreduce_grad_dtype, zero_fill, stream)

    def _prepare_allreduce_pack_buffer(self, allreduce_grad_dtype, n_elems):
        if False:
            return 10
        allreduce_grad_n_bytes = allreduce_grad_dtype.itemsize * n_elems
        needs_sync = False
        if self.gpu_buffer_a.size != allreduce_grad_n_bytes:
            self.gpu_buffer_a.assign(allreduce_grad_n_bytes)
            needs_sync = True
        if self.gpu_buffer_b.size != allreduce_grad_n_bytes:
            self.gpu_buffer_b.assign(allreduce_grad_n_bytes)
            needs_sync = True
        return needs_sync

    def _multi_node_mean_nccl(self, sendbuf, recvbuf, n_elems, dtype, stream=None):
        if False:
            for i in range(10):
                print('nop')
        'Compute mean of each element on each processes with NCCL.\n\n        The function compute mean of each element in ``sendbuf`` on each\n        processes. The result is stored in ``recvbuf``. NCCL is used for\n        communication.\n\n        Args:\n            sendbuf (numpy/cupy array): Input arrays.\n            recvbuf (numpy/cupy array): Output arrays.\n            n_elems (int): the number of elements in `sendbuf`.\n            dtype: Data type of elements used in All-Reduce.\n            stream: CUDA stream used for All-Reduce.\n\n        '
        if chainer.is_debug():
            stream.synchronize()
            array_a = sendbuf.array(n_elems, dtype=dtype)
            array_b = recvbuf.array(n_elems, dtype=dtype)
            self._check_ready_to_allreduce(array_a, array_b)
        if stream is None:
            stream = chainer.cuda.Stream.null
        self._init_comms()
        type_id = _communication_utility._get_nccl_type_id(dtype)
        self.nccl_comm.allReduce(sendbuf.ptr(), recvbuf.ptr(), n_elems, type_id, nccl.NCCL_SUM, stream.ptr)
        div_by_size = chainer.cuda.elementwise('', '{} x'.format(dtype.name), 'x *= (1.0/{})'.format(self.size), 'div_by_size')
        div_by_size(recvbuf.array(n_elems, dtype=dtype), stream=stream)
        if chainer.is_debug():
            stream.synchronize()
            self._ensure_all_finite(recvbuf.array(n_elems, dtype=dtype))