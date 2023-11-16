from numba import cuda
from numpy import array as np_array
from numba.np.ufunc import deviceufunc
from numba.np.ufunc.deviceufunc import UFuncMechanism, GeneralizedUFunc, GUFuncCallSteps

class CUDAUFuncDispatcher(object):
    """
    Invoke the CUDA ufunc specialization for the given inputs.
    """

    def __init__(self, types_to_retty_kernels, pyfunc):
        if False:
            i = 10
            return i + 15
        self.functions = types_to_retty_kernels
        self.__name__ = pyfunc.__name__

    def __call__(self, *args, **kws):
        if False:
            for i in range(10):
                print('nop')
        '\n        *args: numpy arrays or DeviceArrayBase (created by cuda.to_device).\n               Cannot mix the two types in one call.\n\n        **kws:\n            stream -- cuda stream; when defined, asynchronous mode is used.\n            out    -- output array. Can be a numpy array or DeviceArrayBase\n                      depending on the input arguments.  Type must match\n                      the input arguments.\n        '
        return CUDAUFuncMechanism.call(self.functions, args, kws)

    def reduce(self, arg, stream=0):
        if False:
            while True:
                i = 10
        assert len(list(self.functions.keys())[0]) == 2, 'must be a binary ufunc'
        assert arg.ndim == 1, 'must use 1d array'
        n = arg.shape[0]
        gpu_mems = []
        if n == 0:
            raise TypeError('Reduction on an empty array.')
        elif n == 1:
            return arg[0]
        stream = stream or cuda.stream()
        with stream.auto_synchronize():
            if cuda.cudadrv.devicearray.is_cuda_ndarray(arg):
                mem = arg
            else:
                mem = cuda.to_device(arg, stream)
            out = self.__reduce(mem, gpu_mems, stream)
            buf = np_array((1,), dtype=arg.dtype)
            out.copy_to_host(buf, stream=stream)
        return buf[0]

    def __reduce(self, mem, gpu_mems, stream):
        if False:
            print('Hello World!')
        n = mem.shape[0]
        if n % 2 != 0:
            (fatcut, thincut) = mem.split(n - 1)
            gpu_mems.append(fatcut)
            gpu_mems.append(thincut)
            out = self.__reduce(fatcut, gpu_mems, stream)
            gpu_mems.append(out)
            return self(out, thincut, out=out, stream=stream)
        else:
            (left, right) = mem.split(n // 2)
            gpu_mems.append(left)
            gpu_mems.append(right)
            self(left, right, out=left, stream=stream)
            if n // 2 > 1:
                return self.__reduce(left, gpu_mems, stream)
            else:
                return left

class _CUDAGUFuncCallSteps(GUFuncCallSteps):
    __slots__ = ['_stream']

    def __init__(self, nin, nout, args, kwargs):
        if False:
            return 10
        super().__init__(nin, nout, args, kwargs)
        self._stream = kwargs.get('stream', 0)

    def is_device_array(self, obj):
        if False:
            for i in range(10):
                print('nop')
        return cuda.is_cuda_array(obj)

    def as_device_array(self, obj):
        if False:
            print('Hello World!')
        if cuda.cudadrv.devicearray.is_cuda_ndarray(obj):
            return obj
        return cuda.as_cuda_array(obj)

    def to_device(self, hostary):
        if False:
            for i in range(10):
                print('nop')
        return cuda.to_device(hostary, stream=self._stream)

    def to_host(self, devary, hostary):
        if False:
            while True:
                i = 10
        out = devary.copy_to_host(hostary, stream=self._stream)
        return out

    def allocate_device_array(self, shape, dtype):
        if False:
            for i in range(10):
                print('nop')
        return cuda.device_array(shape=shape, dtype=dtype, stream=self._stream)

    def launch_kernel(self, kernel, nelem, args):
        if False:
            while True:
                i = 10
        kernel.forall(nelem, stream=self._stream)(*args)

class CUDAGeneralizedUFunc(GeneralizedUFunc):

    def __init__(self, kernelmap, engine, pyfunc):
        if False:
            for i in range(10):
                print('nop')
        self.__name__ = pyfunc.__name__
        super().__init__(kernelmap, engine)

    @property
    def _call_steps(self):
        if False:
            i = 10
            return i + 15
        return _CUDAGUFuncCallSteps

    def _broadcast_scalar_input(self, ary, shape):
        if False:
            for i in range(10):
                print('nop')
        return cuda.cudadrv.devicearray.DeviceNDArray(shape=shape, strides=(0,), dtype=ary.dtype, gpu_data=ary.gpu_data)

    def _broadcast_add_axis(self, ary, newshape):
        if False:
            i = 10
            return i + 15
        newax = len(newshape) - len(ary.shape)
        newstrides = (0,) * newax + ary.strides
        return cuda.cudadrv.devicearray.DeviceNDArray(shape=newshape, strides=newstrides, dtype=ary.dtype, gpu_data=ary.gpu_data)

class CUDAUFuncMechanism(UFuncMechanism):
    """
    Provide CUDA specialization
    """
    DEFAULT_STREAM = 0

    def launch(self, func, count, stream, args):
        if False:
            while True:
                i = 10
        func.forall(count, stream=stream)(*args)

    def is_device_array(self, obj):
        if False:
            i = 10
            return i + 15
        return cuda.is_cuda_array(obj)

    def as_device_array(self, obj):
        if False:
            print('Hello World!')
        if cuda.cudadrv.devicearray.is_cuda_ndarray(obj):
            return obj
        return cuda.as_cuda_array(obj)

    def to_device(self, hostary, stream):
        if False:
            print('Hello World!')
        return cuda.to_device(hostary, stream=stream)

    def to_host(self, devary, stream):
        if False:
            for i in range(10):
                print('nop')
        return devary.copy_to_host(stream=stream)

    def allocate_device_array(self, shape, dtype, stream):
        if False:
            for i in range(10):
                print('nop')
        return cuda.device_array(shape=shape, dtype=dtype, stream=stream)

    def broadcast_device(self, ary, shape):
        if False:
            while True:
                i = 10
        ax_differs = [ax for ax in range(len(shape)) if ax >= ary.ndim or ary.shape[ax] != shape[ax]]
        missingdim = len(shape) - len(ary.shape)
        strides = [0] * missingdim + list(ary.strides)
        for ax in ax_differs:
            strides[ax] = 0
        return cuda.cudadrv.devicearray.DeviceNDArray(shape=shape, strides=strides, dtype=ary.dtype, gpu_data=ary.gpu_data)
vectorizer_stager_source = '\ndef __vectorized_{name}({args}, __out__):\n    __tid__ = __cuda__.grid(1)\n    if __tid__ < __out__.shape[0]:\n        __out__[__tid__] = __core__({argitems})\n'

class CUDAVectorize(deviceufunc.DeviceVectorize):

    def _compile_core(self, sig):
        if False:
            for i in range(10):
                print('nop')
        cudevfn = cuda.jit(sig, device=True, inline=True)(self.pyfunc)
        return (cudevfn, cudevfn.overloads[sig.args].signature.return_type)

    def _get_globals(self, corefn):
        if False:
            for i in range(10):
                print('nop')
        glbl = self.pyfunc.__globals__.copy()
        glbl.update({'__cuda__': cuda, '__core__': corefn})
        return glbl

    def _compile_kernel(self, fnobj, sig):
        if False:
            print('Hello World!')
        return cuda.jit(fnobj)

    def build_ufunc(self):
        if False:
            for i in range(10):
                print('nop')
        return CUDAUFuncDispatcher(self.kernelmap, self.pyfunc)

    @property
    def _kernel_template(self):
        if False:
            while True:
                i = 10
        return vectorizer_stager_source
_gufunc_stager_source = '\ndef __gufunc_{name}({args}):\n    __tid__ = __cuda__.grid(1)\n    if __tid__ < {checkedarg}:\n        __core__({argitems})\n'

class CUDAGUFuncVectorize(deviceufunc.DeviceGUFuncVectorize):

    def build_ufunc(self):
        if False:
            i = 10
            return i + 15
        engine = deviceufunc.GUFuncEngine(self.inputsig, self.outputsig)
        return CUDAGeneralizedUFunc(kernelmap=self.kernelmap, engine=engine, pyfunc=self.pyfunc)

    def _compile_kernel(self, fnobj, sig):
        if False:
            while True:
                i = 10
        return cuda.jit(sig)(fnobj)

    @property
    def _kernel_template(self):
        if False:
            print('Hello World!')
        return _gufunc_stager_source

    def _get_globals(self, sig):
        if False:
            return 10
        corefn = cuda.jit(sig, device=True)(self.pyfunc)
        glbls = self.py_func.__globals__.copy()
        glbls.update({'__cuda__': cuda, '__core__': corefn})
        return glbls