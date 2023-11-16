from __future__ import absolute_import
import abc
import sys
import threading
import typing as tp
import warnings
import numpy
import chainer
from chainer import backend
from chainer.backends import _cpu
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import types
from chainer import utils
import chainerx
_thread_local = threading.local()
_thread_local.flag_gpu_to_gpu = None

class DeviceResident(utils.enable_final(meta_base=abc.ABCMeta)):
    """A base class of objects with multi-device hierarchy."""
    _device = _cpu.CpuDevice()

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._overridden_to_methods = tuple([m for m in ('to_cpu', 'to_gpu', 'to_intel64') if _is_to_device_method_overridden(self, m)])

    def device_resident_accept(self, visitor):
        if False:
            return 10
        'Applies the visitor to all the device objects in this instance.\n\n        Args:\n            visitor(~chainer.device_resident.DeviceResidentsVisitor): Visitor.\n\n        This method should be overridden if the concrete class has custom\n        sub-hierarchy of device resident objects.\n        '
        visitor.visit_device_resident(self)

    @property
    def device(self):
        if False:
            i = 10
            return i + 15
        ':class:`~chainer.backend.Device` instance.'
        return self._device

    @property
    def xp(self) -> types.Xp:
        if False:
            while True:
                i = 10
        'Array module corresponding to the device.\n\n        Depending on the device in which this object resides, this property\n        returns :mod:`numpy`, :mod:`cupy` or :mod:`chainerx`.\n\n        '
        device = self.device
        if device is None:
            return None
        return device.xp

    @utils.final(action=DeprecationWarning)
    def to_cpu(self) -> 'DeviceResident':
        if False:
            while True:
                i = 10
        'Copies parameter variables and persistent values to CPU.\n\n         .. deprecated:: v7.0.0\n            Use :meth:`to_device` instead.\n\n        This method does not handle non-registered attributes. If some of such\n        attributes must be copied to CPU, the link implementation should\n        override :meth:`~DeviceResident.device_resident_accept` to do so.\n\n        Returns: self\n\n        '
        visitor = _ToDeviceVisitor(backend.CpuDevice(), entry_method_info=('to_cpu', {}), starting_device_resident=self)
        self.__to_device(visitor)
        return self

    @utils.final(action=DeprecationWarning)
    def to_gpu(self, device: tp.Optional[types.CudaDeviceSpec]=None) -> 'DeviceResident':
        if False:
            return 10
        'Copies parameter variables and persistent values to GPU.\n\n         .. deprecated:: v7.0.0\n            Use :meth:`to_device` instead.\n\n        This method does not handle non-registered attributes. If some of such\n        attributes must be copied to GPU, the link implementation must\n        override :meth:`~DeviceResident.device_resident_accept` to do so.\n\n        .. warning::\n\n            This method does not transfer the parameters if they are already on\n            GPU. Use ``to_device`` to perform inter-GPU transfer.\n\n        Args:\n            device: Target device specifier. If omitted, the current device is\n                used.\n\n        Returns: self\n\n        '
        cuda.check_cuda_available()
        cuda_device = cuda._get_device_or_current(device)
        device = chainer.backends.cuda.GpuDevice(cuda_device)
        visitor = _ToDeviceVisitor(device, entry_method_info=('to_gpu', {'device': device.device}), skip_between_cupy_devices=True, starting_device_resident=self)
        self.__to_device(visitor)
        return self

    @utils.final(action=DeprecationWarning)
    def to_intel64(self) -> 'DeviceResident':
        if False:
            while True:
                i = 10
        'Copies parameter variables and persistent values to CPU.\n\n         .. deprecated:: v7.0.0\n            Use :meth:`to_device` instead.\n\n        '
        intel64.check_ideep_available()
        visitor = _ToDeviceVisitor(chainer.get_device(intel64.Intel64Device()), entry_method_info=('to_intel64', {}), starting_device_resident=self)
        self.__to_device(visitor)
        return self

    @utils.final
    def to_chx(self):
        if False:
            return 10
        'Converts parameter variables and persistent values to ChainerX without any copy.\n\n        This method does not handle non-registered attributes. If some of such\n        attributes must be copied to ChainerX, the link implementation must\n        override this method to do so.\n\n        Returns: self\n        '
        if not chainerx.is_available():
            raise RuntimeError('ChainerX is not available.')
        if self.xp is chainerx:
            return self
        self.device_resident_accept(_ToChxVisitor())
        return self

    @utils.final
    def from_chx(self):
        if False:
            while True:
                i = 10
        'Converts parameter variables and persistent values from ChainerX to NumPy/CuPy devices without any copy.'
        if self._device.xp is chainerx:
            self._device = self._device.fallback_device
        self.device_resident_accept(_FromChxVisitor())
        return self

    def __to_device(self, to_device_visitor):
        if False:
            print('Hello World!')
        self.device_resident_accept(to_device_visitor)

    @utils.final
    def to_device(self, device: types.DeviceSpec) -> 'DeviceResident':
        if False:
            i = 10
            return i + 15
        'Copies parameter variables and persistent values to the specified device.\n\n        This method does not handle non-registered attributes. If some of such\n        attributes must be copied to the device, the link implementation must\n        override this method to do so.\n\n        Args:\n            device: Target device specifier. See\n                :func:`~chainer.get_device` for available values.\n\n        Returns: self\n\n        '
        device = chainer.get_device(device)
        self.__to_device(_ToDeviceVisitor(device))
        return self

def _is_to_device_method_overridden(device_resident, method_name):
    if False:
        return 10
    to_method = getattr(device_resident, method_name, None).__func__
    to_method_orig = getattr(DeviceResident, method_name)
    if sys.version_info < (3,):
        to_method_orig = to_method_orig.__func__
    if to_method is not to_method_orig:
        return True
    return False

class DeviceResidentsVisitor(object):
    """Base class of visitors that visits device resident objects recursively.

    ..  seealso::
        :class:`chainer.DeviceResident`
    """

    def visit_device_resident(self, device_resident):
        if False:
            i = 10
            return i + 15
        'Processes a :class:`~chainer.DeviceResident` instance.'
        raise NotImplementedError()

    def visit_array(self, arr):
        if False:
            for i in range(10):
                print('nop')
        'Processes an array and returns a new one.\n\n        If the visitor does not create a new array, it can simply return the\n        original array.\n        '
        raise NotImplementedError()

    def visit_variable(self, param):
        if False:
            print('Hello World!')
        'Processes a :class:`~chainer.Variable` or a :class:`~chainer.Parameter`.'
        raise NotImplementedError()

class _ToDeviceVisitor(DeviceResidentsVisitor):

    def __init__(self, device, entry_method_info=None, skip_between_cupy_devices=False, starting_device_resident=None):
        if False:
            print('Hello World!')
        assert isinstance(device, chainer.backend.Device)
        if entry_method_info is not None:
            device_names = {'to_cpu': '@numpy', 'to_gpu': '@cupy:N', 'to_intel64': '@intel64'}
            assert len(entry_method_info) == 2
            method = entry_method_info[0]
            assert method in device_names
            warnings.warn("{} is deprecated. Please use to_device('{}') instead.".format(method, device_names[method]), DeprecationWarning)
        assert starting_device_resident is None or isinstance(starting_device_resident, DeviceResident)
        self._device = device
        self._entry_method_info = entry_method_info
        self._skip_between_cupy_devices = skip_between_cupy_devices
        self._starting_device_resident = starting_device_resident

    def visit_device_resident(self, device_resident):
        if False:
            i = 10
            return i + 15
        device_resident._device = self._device
        if device_resident._overridden_to_methods:
            if device_resident is self._starting_device_resident:
                return
            if self._entry_method_info is not None:
                (method_name, kwargs) = self._entry_method_info
            else:
                (method_name, kwargs) = self._device_to_method_name_and_kwargs(self._device)
            if method_name in device_resident._overridden_to_methods:
                to_method = getattr(device_resident, method_name)
                to_method(**kwargs)
                return

    def _device_to_method_name_and_kwargs(self, device):
        if False:
            return 10
        if device.xp is chainerx:
            return (None, {})
        if device.xp is cuda.cupy:
            return ('to_gpu', {'device': device.device.id})
        assert device.xp is numpy
        if isinstance(device, _cpu.CpuDevice):
            return ('to_cpu', {})
        assert isinstance(device, intel64.Intel64Device)
        return ('to_intel64', {})

    def visit_array(self, arr):
        if False:
            while True:
                i = 10
        assert isinstance(arr, chainer.get_array_types())
        device = backend.get_device_from_array(arr)
        if self._skip_visiting(device):
            self._warn_to_gpu(device, self._device)
            return arr
        return self._device.send(arr)

    def visit_variable(self, param):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(param, chainer.Variable)
        device = param.device
        if self._skip_visiting(device):
            self._warn_to_gpu(device, self._device)
            return
        param.to_device(self._device)

    def _skip_visiting(self, obj_device):
        if False:
            while True:
                i = 10
        return self._skip_between_cupy_devices and isinstance(self._device, backend.GpuDevice) and isinstance(obj_device, backend.GpuDevice)

    @staticmethod
    def _warn_to_gpu(src_device, dst_device):
        if False:
            while True:
                i = 10
        src_id = src_device.device.id
        dst_id = dst_device.device.id
        if src_id != dst_id:
            if _thread_local.flag_gpu_to_gpu is None:
                warnings.warn('You are trying to transfer a DeviceResident to GPU-{dst} which is already on GPU-{src}.\n`DeviceResident.to_gpu` does nothing if the DeviceResident is already on GPU.\n\nYou can use `DeviceResident.to_device()` method to perform inter-GPU transfer.\n'.format(dst=dst_id, src=src_id), RuntimeWarning)
            else:
                assert isinstance(_thread_local.flag_gpu_to_gpu, bool)
                _thread_local.flag_gpu_to_gpu = True

class _ToChxVisitor(DeviceResidentsVisitor):

    def visit_device_resident(self, device_resident):
        if False:
            i = 10
            return i + 15
        device_resident._device = backend.ChainerxDevice.from_fallback_device(device_resident._device)

    def visit_array(self, arr):
        if False:
            print('Hello World!')
        assert isinstance(arr, chainer.get_array_types())
        return backend.to_chx(arr)

    def visit_variable(self, param):
        if False:
            print('Hello World!')
        assert isinstance(param, chainer.Variable)
        param.to_chx()

class _FromChxVisitor(DeviceResidentsVisitor):

    def visit_device_resident(self, device_resident):
        if False:
            print('Hello World!')
        if isinstance(device_resident._device, backend.ChainerxDevice):
            device_resident._device = device_resident._device.fallback_device

    def visit_array(self, arr):
        if False:
            return 10
        assert isinstance(arr, chainer.get_array_types())
        return backend.from_chx(arr)

    def visit_variable(self, param):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(param, chainer.Variable)
        param.from_chx()