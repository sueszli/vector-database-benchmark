""" Defines an in-process KernelManager with signals and slots.
"""
from qtpy import QtCore
from ipykernel.inprocess import InProcessHBChannel, InProcessKernelClient, InProcessKernelManager
from ipykernel.inprocess.channels import InProcessChannel
from traitlets import Type
from .util import SuperQObject
from .kernel_mixins import QtKernelClientMixin, QtKernelManagerMixin
from .rich_jupyter_widget import RichJupyterWidget

class QtInProcessChannel(SuperQObject, InProcessChannel):
    started = QtCore.Signal()
    stopped = QtCore.Signal()
    message_received = QtCore.Signal(object)

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        ' Reimplemented to emit signal.\n        '
        super().start()
        self.started.emit()

    def stop(self):
        if False:
            print('Hello World!')
        ' Reimplemented to emit signal.\n        '
        super().stop()
        self.stopped.emit()

    def call_handlers_later(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        ' Call the message handlers later.\n        '
        do_later = lambda : self.call_handlers(*args, **kwds)
        QtCore.QTimer.singleShot(0, do_later)

    def call_handlers(self, msg):
        if False:
            i = 10
            return i + 15
        self.message_received.emit(msg)

    def process_events(self):
        if False:
            for i in range(10):
                print('nop')
        ' Process any pending GUI events.\n        '
        QtCore.QCoreApplication.instance().processEvents()

    def flush(self, timeout=1.0):
        if False:
            i = 10
            return i + 15
        ' Reimplemented to ensure that signals are dispatched immediately.\n        '
        super().flush()
        self.process_events()

    def closed(self):
        if False:
            for i in range(10):
                print('nop')
        ' Function to ensure compatibility with the QtZMQSocketChannel.'
        return False

class QtInProcessHBChannel(SuperQObject, InProcessHBChannel):
    kernel_died = QtCore.Signal()

class QtInProcessKernelClient(QtKernelClientMixin, InProcessKernelClient):
    """ An in-process KernelManager with signals and slots.
    """
    iopub_channel_class = Type(QtInProcessChannel)
    shell_channel_class = Type(QtInProcessChannel)
    stdin_channel_class = Type(QtInProcessChannel)
    hb_channel_class = Type(QtInProcessHBChannel)

class QtInProcessKernelManager(QtKernelManagerMixin, InProcessKernelManager):
    client_class = __module__ + '.QtInProcessKernelClient'

class QtInProcessRichJupyterWidget(RichJupyterWidget):
    """ An in-process Jupyter Widget that enables multiline editing
    """

    def _is_complete(self, source, interactive=True):
        if False:
            return 10
        shell = self.kernel_manager.kernel.shell
        (status, indent_spaces) = shell.input_transformer_manager.check_complete(source)
        if indent_spaces is None:
            indent = ''
        else:
            indent = ' ' * indent_spaces
        return (status != 'incomplete', indent)