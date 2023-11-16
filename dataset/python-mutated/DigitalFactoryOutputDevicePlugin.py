from UM.OutputDevice.OutputDevicePlugin import OutputDevicePlugin
from .DigitalFactoryOutputDevice import DigitalFactoryOutputDevice
from .DigitalFactoryController import DigitalFactoryController

class DigitalFactoryOutputDevicePlugin(OutputDevicePlugin):

    def __init__(self, df_controller: DigitalFactoryController) -> None:
        if False:
            return 10
        super().__init__()
        self.df_controller = df_controller

    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        self.getOutputDeviceManager().addProjectOutputDevice(DigitalFactoryOutputDevice(plugin_id=self.getPluginId(), df_controller=self.df_controller, add_to_output_devices=True))

    def stop(self) -> None:
        if False:
            return 10
        self.getOutputDeviceManager().removeProjectOutputDevice('digital_factory')