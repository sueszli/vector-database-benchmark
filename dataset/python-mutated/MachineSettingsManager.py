from typing import Optional, TYPE_CHECKING
from PyQt6.QtCore import QObject, pyqtSlot
from UM.i18n import i18nCatalog
from cura.Machines.ContainerTree import ContainerTree
if TYPE_CHECKING:
    from cura.CuraApplication import CuraApplication

class MachineSettingsManager(QObject):

    def __init__(self, application: 'CuraApplication', parent: Optional['QObject']=None) -> None:
        if False:
            return 10
        super().__init__(parent)
        self._i18n_catalog = i18nCatalog('cura')
        self._application = application

    @pyqtSlot()
    def forceUpdate(self) -> None:
        if False:
            print('Hello World!')
        self._application.getMachineManager().globalContainerChanged.emit()

    @pyqtSlot(int)
    def updateMaterialForDiameter(self, extruder_position: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._application.getMachineManager().updateMaterialWithVariant(str(extruder_position))

    @pyqtSlot(int)
    def setMachineExtruderCount(self, extruder_count: int) -> None:
        if False:
            print('Hello World!')
        self._application.getMachineManager().setActiveMachineExtruderCount(extruder_count)

    @pyqtSlot()
    def updateHasMaterialsMetadata(self):
        if False:
            return 10
        machine_manager = self._application.getMachineManager()
        global_stack = machine_manager.activeMachine
        definition = global_stack.definition
        if definition.getProperty('machine_gcode_flavor', 'value') != 'UltiGCode' or definition.getMetaDataEntry('has_materials', False):
            return
        has_materials = global_stack.getProperty('machine_gcode_flavor', 'value') != 'UltiGCode'
        material_node = None
        if has_materials:
            global_stack.setMetaDataEntry('has_materials', True)
        elif 'has_materials' in global_stack.getMetaData():
            global_stack.removeMetaDataEntry('has_materials')
        for (position, extruder) in enumerate(global_stack.extruderList):
            if has_materials:
                approximate_diameter = extruder.getApproximateMaterialDiameter()
                variant_node = ContainerTree.getInstance().machines[global_stack.definition.getId()].variants[extruder.variant.getName()]
                material_node = variant_node.preferredMaterial(approximate_diameter)
            machine_manager.setMaterial(str(position), material_node)
        self.forceUpdate()