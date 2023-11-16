from UM.Settings.ContainerRegistry import ContainerRegistry
from cura.MachineAction import MachineAction
from PyQt6.QtCore import pyqtSlot, pyqtSignal, pyqtProperty
from UM.i18n import i18nCatalog
from UM.Application import Application
catalog = i18nCatalog('cura')
from cura.Settings.CuraStackBuilder import CuraStackBuilder

class UMOUpgradeSelection(MachineAction):
    """The Ultimaker Original can have a few revisions & upgrades.
    This action helps with selecting them, so they are added as a variant.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('UMOUpgradeSelection', catalog.i18nc('@action', 'Select upgrades'))
        self._qml_url = 'UMOUpgradeSelectionMachineAction.qml'

    def _reset(self):
        if False:
            return 10
        self.heatedBedChanged.emit()
    heatedBedChanged = pyqtSignal()

    @pyqtProperty(bool, notify=heatedBedChanged)
    def hasHeatedBed(self):
        if False:
            print('Hello World!')
        global_container_stack = Application.getInstance().getGlobalContainerStack()
        if global_container_stack:
            return global_container_stack.getProperty('machine_heated_bed', 'value')

    @pyqtSlot(bool)
    def setHeatedBed(self, heated_bed=True):
        if False:
            while True:
                i = 10
        global_container_stack = Application.getInstance().getGlobalContainerStack()
        if global_container_stack:
            definition_changes_container = global_container_stack.definitionChanges
            if definition_changes_container == ContainerRegistry.getInstance().getEmptyInstanceContainer():
                definition_changes_container = CuraStackBuilder.createDefinitionChangesContainer(global_container_stack, global_container_stack.getId() + '_settings')
            definition_changes_container.setProperty('machine_heated_bed', 'value', heated_bed)
            self.heatedBedChanged.emit()