from UM.Job import Job
from UM.Application import Application

class ProcessGCodeLayerJob(Job):

    def __init__(self, message):
        if False:
            print('Hello World!')
        super().__init__()
        self._scene = Application.getInstance().getController().getScene()
        self._message = message

    def run(self):
        if False:
            i = 10
            return i + 15
        active_build_plate_id = Application.getInstance().getMultiBuildPlateModel().activeBuildPlate
        gcode_list = self._scene.gcode_dict[active_build_plate_id]
        gcode_list.append(self._message.data.decode('utf-8', 'replace'))