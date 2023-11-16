import os
import threading
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from UM.FlameProfiler import pyqtSlot
from UM.Application import Application
from UM.PluginRegistry import PluginRegistry
from UM.Logger import Logger
from UM.i18n import i18nCatalog
catalog = i18nCatalog('cura')

class ImageReaderUI(QObject):
    show_config_ui_trigger = pyqtSignal()

    def __init__(self, image_reader):
        if False:
            while True:
                i = 10
        super(ImageReaderUI, self).__init__()
        self.image_reader = image_reader
        self._ui_view = None
        self.show_config_ui_trigger.connect(self._actualShowConfigUI)
        self.default_width = 120
        self.default_depth = 120
        self._aspect = 1
        self._width = self.default_width
        self._depth = self.default_depth
        self.base_height = 0.4
        self.peak_height = 2.5
        self.smoothing = 1
        self.lighter_is_higher = False
        self.use_transparency_model = True
        self.transmittance_1mm = 50.0
        self._ui_lock = threading.Lock()
        self._cancelled = False
        self._disable_size_callbacks = False

    def setWidthAndDepth(self, width, depth):
        if False:
            i = 10
            return i + 15
        self._aspect = width / depth
        self._width = width
        self._depth = depth

    def getWidth(self):
        if False:
            i = 10
            return i + 15
        return self._width

    def getDepth(self):
        if False:
            print('Hello World!')
        return self._depth

    def getCancelled(self):
        if False:
            for i in range(10):
                print('nop')
        return self._cancelled

    def waitForUIToClose(self):
        if False:
            for i in range(10):
                print('nop')
        self._ui_lock.acquire()
        self._ui_lock.release()

    def showConfigUI(self):
        if False:
            return 10
        self._ui_lock.acquire()
        self._cancelled = False
        self.show_config_ui_trigger.emit()

    def _actualShowConfigUI(self):
        if False:
            print('Hello World!')
        self._disable_size_callbacks = True
        if self._ui_view is None:
            self._createConfigUI()
        self._ui_view.show()
        self._ui_view.findChild(QObject, 'Width').setProperty('text', str(self._width))
        self._ui_view.findChild(QObject, 'Depth').setProperty('text', str(self._depth))
        self._disable_size_callbacks = False
        self._ui_view.findChild(QObject, 'Base_Height').setProperty('text', str(self.base_height))
        self._ui_view.findChild(QObject, 'Peak_Height').setProperty('text', str(self.peak_height))
        self._ui_view.findChild(QObject, 'Transmittance').setProperty('text', str(self.transmittance_1mm))
        self._ui_view.findChild(QObject, 'Smoothing').setProperty('value', self.smoothing)

    def _createConfigUI(self):
        if False:
            while True:
                i = 10
        if self._ui_view is None:
            Logger.log('d', 'Creating ImageReader config UI')
            path = os.path.join(PluginRegistry.getInstance().getPluginPath('ImageReader'), 'ConfigUI.qml')
            self._ui_view = Application.getInstance().createQmlComponent(path, {'manager': self})
            self._ui_view.setFlags(self._ui_view.flags() & ~Qt.WindowType.WindowCloseButtonHint & ~Qt.WindowType.WindowMinimizeButtonHint & ~Qt.WindowType.WindowMaximizeButtonHint)
            self._disable_size_callbacks = False

    @pyqtSlot()
    def onOkButtonClicked(self):
        if False:
            for i in range(10):
                print('nop')
        self._cancelled = False
        self._ui_view.close()
        try:
            self._ui_lock.release()
        except RuntimeError:
            pass

    @pyqtSlot()
    def onCancelButtonClicked(self):
        if False:
            i = 10
            return i + 15
        self._cancelled = True
        self._ui_view.close()
        try:
            self._ui_lock.release()
        except RuntimeError:
            pass

    @pyqtSlot(str)
    def onWidthChanged(self, value):
        if False:
            while True:
                i = 10
        if self._ui_view and (not self._disable_size_callbacks):
            if len(value) > 0:
                try:
                    self._width = float(value.replace(',', '.'))
                except ValueError:
                    self._width = 0
            else:
                self._width = 0
            self._depth = self._width / self._aspect
            self._disable_size_callbacks = True
            self._ui_view.findChild(QObject, 'Depth').setProperty('text', str(self._depth))
            self._disable_size_callbacks = False

    @pyqtSlot(str)
    def onDepthChanged(self, value):
        if False:
            i = 10
            return i + 15
        if self._ui_view and (not self._disable_size_callbacks):
            if len(value) > 0:
                try:
                    self._depth = float(value.replace(',', '.'))
                except ValueError:
                    self._depth = 0
            else:
                self._depth = 0
            self._width = self._depth * self._aspect
            self._disable_size_callbacks = True
            self._ui_view.findChild(QObject, 'Width').setProperty('text', str(self._width))
            self._disable_size_callbacks = False

    @pyqtSlot(str)
    def onBaseHeightChanged(self, value):
        if False:
            print('Hello World!')
        if len(value) > 0:
            try:
                self.base_height = float(value.replace(',', '.'))
            except ValueError:
                self.base_height = 0
        else:
            self.base_height = 0

    @pyqtSlot(str)
    def onPeakHeightChanged(self, value):
        if False:
            while True:
                i = 10
        if len(value) > 0:
            try:
                self.peak_height = float(value.replace(',', '.'))
                if self.peak_height < 0:
                    self.peak_height = 2.5
            except ValueError:
                self.peak_height = 2.5
        else:
            self.peak_height = 0

    @pyqtSlot(float)
    def onSmoothingChanged(self, value):
        if False:
            print('Hello World!')
        self.smoothing = int(value)

    @pyqtSlot(int)
    def onImageColorInvertChanged(self, value):
        if False:
            print('Hello World!')
        self.lighter_is_higher = value == 1

    @pyqtSlot(int)
    def onColorModelChanged(self, value):
        if False:
            print('Hello World!')
        self.use_transparency_model = value == 1

    @pyqtSlot(int)
    def onTransmittanceChanged(self, value):
        if False:
            i = 10
            return i + 15
        self.transmittance_1mm = value