from cura.PrinterOutput.Models.PrintJobOutputModel import PrintJobOutputModel
from cura.PrinterOutput.PrinterOutputController import PrinterOutputController
from cura.PrinterOutput.PrinterOutputDevice import PrinterOutputDevice

class ClusterOutputController(PrinterOutputController):

    def __init__(self, output_device: PrinterOutputDevice) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(output_device)
        self.can_pause = True
        self.can_abort = True
        self.can_pre_heat_bed = False
        self.can_pre_heat_hotends = False
        self.can_send_raw_gcode = False
        self.can_control_manually = False
        self.can_update_firmware = False

    def setJobState(self, job: PrintJobOutputModel, state: str):
        if False:
            while True:
                i = 10
        self._output_device.setJobState(job.key, state)