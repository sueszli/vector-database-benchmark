"""Defines a KernelManager that provides signals and slots."""
from qtpy import QtCore
from traitlets import HasTraits, Type
from .util import MetaQObjectHasTraits, SuperQObject
from .comms import CommManager

class QtKernelRestarterMixin(MetaQObjectHasTraits('NewBase', (HasTraits, SuperQObject), {})):
    _timer = None

class QtKernelManagerMixin(MetaQObjectHasTraits('NewBase', (HasTraits, SuperQObject), {})):
    """ A KernelClient that provides signals and slots.
    """
    kernel_restarted = QtCore.Signal()

class QtKernelClientMixin(MetaQObjectHasTraits('NewBase', (HasTraits, SuperQObject), {})):
    """ A KernelClient that provides signals and slots.
    """
    started_channels = QtCore.Signal()
    stopped_channels = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.comm_manager = None

    def start_channels(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        ' Reimplemented to emit signal.\n        '
        super().start_channels(*args, **kw)
        self.started_channels.emit()
        self.comm_manager = CommManager(parent=self, kernel_client=self)

    def stop_channels(self):
        if False:
            print('Hello World!')
        ' Reimplemented to emit signal.\n        '
        super().stop_channels()
        self.stopped_channels.emit()
        self.comm_manager = None