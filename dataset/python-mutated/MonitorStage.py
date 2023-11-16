import os.path
from UM.Application import Application
from cura.Stages.CuraStage import CuraStage

class MonitorStage(CuraStage):
    """Stage for monitoring a 3D printing while it's printing."""

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        Application.getInstance().engineCreatedSignal.connect(self._onEngineCreated)
        self._printer_output_device = None
        self._active_print_job = None
        self._active_printer = None

    def _setActivePrintJob(self, print_job):
        if False:
            return 10
        if self._active_print_job != print_job:
            self._active_print_job = print_job

    def _setActivePrinter(self, printer):
        if False:
            for i in range(10):
                print('nop')
        if self._active_printer != printer:
            if self._active_printer:
                self._active_printer.activePrintJobChanged.disconnect(self._onActivePrintJobChanged)
            self._active_printer = printer
            if self._active_printer:
                self._setActivePrintJob(self._active_printer.activePrintJob)
                self._active_printer.activePrintJobChanged.connect(self._onActivePrintJobChanged)
            else:
                self._setActivePrintJob(None)

    def _onActivePrintJobChanged(self):
        if False:
            return 10
        self._setActivePrintJob(self._active_printer.activePrintJob)

    def _onActivePrinterChanged(self):
        if False:
            for i in range(10):
                print('nop')
        self._setActivePrinter(self._printer_output_device.activePrinter)

    def _onOutputDevicesChanged(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            new_output_device = Application.getInstance().getMachineManager().printerOutputDevices[0]
            if new_output_device != self._printer_output_device:
                if self._printer_output_device:
                    try:
                        self._printer_output_device.printersChanged.disconnect(self._onActivePrinterChanged)
                    except TypeError:
                        pass
                self._printer_output_device = new_output_device
                self._printer_output_device.printersChanged.connect(self._onActivePrinterChanged)
                self._setActivePrinter(self._printer_output_device.activePrinter)
        except IndexError:
            pass

    def _onEngineCreated(self):
        if False:
            i = 10
            return i + 15
        Application.getInstance().getMachineManager().outputDevicesChanged.connect(self._onOutputDevicesChanged)
        self._onOutputDevicesChanged()
        plugin_path = Application.getInstance().getPluginRegistry().getPluginPath(self.getPluginId())
        if plugin_path is not None:
            menu_component_path = os.path.join(plugin_path, 'MonitorMenu.qml')
            main_component_path = os.path.join(plugin_path, 'MonitorMain.qml')
            self.addDisplayComponent('menu', menu_component_path)
            self.addDisplayComponent('main', main_component_path)