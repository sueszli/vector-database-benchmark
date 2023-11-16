import copy
import uuid
from UM.Scene.SceneNodeDecorator import SceneNodeDecorator
from UM.Signal import Signal, signalemitter
from UM.Settings.InstanceContainer import InstanceContainer
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Logger import Logger
from UM.Application import Application
from cura.Settings.PerObjectContainerStack import PerObjectContainerStack
from cura.Settings.ExtruderManager import ExtruderManager

@signalemitter
class SettingOverrideDecorator(SceneNodeDecorator):
    """A decorator that adds a container stack to a Node. This stack should be queried for all settings regarding

    the linked node. The Stack in question will refer to the global stack (so that settings that are not defined by
    this stack still resolve.
    """
    activeExtruderChanged = Signal()
    'Event indicating that the user selected a different extruder.'
    _non_printing_mesh_settings = {'anti_overhang_mesh', 'infill_mesh', 'cutting_mesh'}
    'Non-printing meshes\n\n    If these settings are True for any mesh, the mesh does not need a convex hull,\n    and is sent to the slicer regardless of whether it fits inside the build volume.\n    Note that Support Mesh is not in here because it actually generates\n    g-code in the volume of the mesh.\n    '
    _non_thumbnail_visible_settings = {'anti_overhang_mesh', 'infill_mesh', 'cutting_mesh', 'support_mesh'}

    def __init__(self, *, force_update=True):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._stack = PerObjectContainerStack(container_id='per_object_stack_' + str(id(self)))
        self._stack.setDirty(False)
        user_container = InstanceContainer(container_id=self._generateUniqueName())
        user_container.setMetaDataEntry('type', 'user')
        self._stack.userChanges = user_container
        self._extruder_stack = ExtruderManager.getInstance().getExtruderStack(0).getId()
        self._is_non_printing_mesh = False
        self._is_non_thumbnail_visible_mesh = False
        self._is_support_mesh = False
        self._is_cutting_mesh = False
        self._is_infill_mesh = False
        self._is_anti_overhang_mesh = False
        self._stack.propertyChanged.connect(self._onSettingChanged)
        Application.getInstance().getContainerRegistry().addContainer(self._stack)
        Application.getInstance().globalContainerStackChanged.connect(self._updateNextStack)
        self.activeExtruderChanged.connect(self._updateNextStack)
        if force_update:
            self._updateNextStack()

    def _generateUniqueName(self):
        if False:
            while True:
                i = 10
        return 'SettingOverrideInstanceContainer-%s' % uuid.uuid1()

    def __deepcopy__(self, memo):
        if False:
            i = 10
            return i + 15
        deep_copy = SettingOverrideDecorator(force_update=False)
        'Create a fresh decorator object'
        instance_container = copy.deepcopy(self._stack.getContainer(0), memo)
        'Copy the instance'
        instance_container.setMetaDataEntry('id', self._generateUniqueName())
        deep_copy._stack.replaceContainer(0, instance_container)
        deep_copy.setActiveExtruder(self._extruder_stack)
        return deep_copy

    def getActiveExtruder(self):
        if False:
            return 10
        "Gets the currently active extruder to print this object with.\n\n        :return: An extruder's container stack.\n        "
        return self._extruder_stack

    def getActiveExtruderChangedSignal(self):
        if False:
            i = 10
            return i + 15
        'Gets the signal that emits if the active extruder changed.\n\n        This can then be accessed via a decorator.\n        '
        return self.activeExtruderChanged

    def getActiveExtruderPosition(self):
        if False:
            while True:
                i = 10
        "Gets the currently active extruders position\n\n        :return: An extruder's position, or None if no position info is available.\n        "
        if self._is_support_mesh:
            global_container_stack = Application.getInstance().getGlobalContainerStack()
            if global_container_stack:
                return str(global_container_stack.getProperty('support_extruder_nr', 'value'))
        containers = ContainerRegistry.getInstance().findContainers(id=self.getActiveExtruder())
        if containers:
            container_stack = containers[0]
            return container_stack.getMetaDataEntry('position', default=None)

    def isCuttingMesh(self):
        if False:
            print('Hello World!')
        return self._is_cutting_mesh

    def isSupportMesh(self):
        if False:
            while True:
                i = 10
        return self._is_support_mesh

    def isInfillMesh(self):
        if False:
            return 10
        return self._is_infill_mesh

    def isAntiOverhangMesh(self):
        if False:
            return 10
        return self._is_anti_overhang_mesh

    def _evaluateAntiOverhangMesh(self):
        if False:
            while True:
                i = 10
        return bool(self._stack.userChanges.getProperty('anti_overhang_mesh', 'value'))

    def _evaluateIsCuttingMesh(self):
        if False:
            while True:
                i = 10
        return bool(self._stack.userChanges.getProperty('cutting_mesh', 'value'))

    def _evaluateIsSupportMesh(self):
        if False:
            return 10
        return bool(self._stack.userChanges.getProperty('support_mesh', 'value'))

    def _evaluateInfillMesh(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self._stack.userChanges.getProperty('infill_mesh', 'value'))

    def isNonPrintingMesh(self):
        if False:
            while True:
                i = 10
        return self._is_non_printing_mesh

    def _evaluateIsNonPrintingMesh(self):
        if False:
            for i in range(10):
                print('nop')
        return any((bool(self._stack.getProperty(setting, 'value')) for setting in self._non_printing_mesh_settings))

    def isNonThumbnailVisibleMesh(self):
        if False:
            i = 10
            return i + 15
        return self._is_non_thumbnail_visible_mesh

    def _evaluateIsNonThumbnailVisibleMesh(self):
        if False:
            for i in range(10):
                print('nop')
        return any((bool(self._stack.getProperty(setting, 'value')) for setting in self._non_thumbnail_visible_settings))

    def _onSettingChanged(self, setting_key, property_name):
        if False:
            return 10
        if property_name == 'value':
            self._is_non_printing_mesh = self._evaluateIsNonPrintingMesh()
            self._is_non_thumbnail_visible_mesh = self._evaluateIsNonThumbnailVisibleMesh()
            if setting_key == 'anti_overhang_mesh':
                self._is_anti_overhang_mesh = self._evaluateAntiOverhangMesh()
            elif setting_key == 'support_mesh':
                self._is_support_mesh = self._evaluateIsSupportMesh()
            elif setting_key == 'cutting_mesh':
                self._is_cutting_mesh = self._evaluateIsCuttingMesh()
            elif setting_key == 'infill_mesh':
                self._is_infill_mesh = self._evaluateInfillMesh()
            Application.getInstance().getBackend().needsSlicing()
            Application.getInstance().getBackend().tickle()

    def _updateNextStack(self):
        if False:
            while True:
                i = 10
        'Makes sure that the stack upon which the container stack is placed is\n\n        kept up to date.\n        '
        if self._extruder_stack:
            extruder_stack = ContainerRegistry.getInstance().findContainerStacks(id=self._extruder_stack)
            if extruder_stack:
                if self._stack.getNextStack():
                    old_extruder_stack_id = self._stack.getNextStack().getId()
                else:
                    old_extruder_stack_id = ''
                self._stack.setNextStack(extruder_stack[0])
                if self._stack.getNextStack().getId() != old_extruder_stack_id:
                    Application.getInstance().getBackend().needsSlicing()
                    Application.getInstance().getBackend().tickle()
            else:
                Logger.log('e', 'Extruder stack %s below per-object settings does not exist.', self._extruder_stack)
        else:
            self._stack.setNextStack(Application.getInstance().getGlobalContainerStack())

    def setActiveExtruder(self, extruder_stack_id):
        if False:
            return 10
        'Changes the extruder with which to print this node.\n\n        :param extruder_stack_id: The new extruder stack to print with.\n        '
        self._extruder_stack = extruder_stack_id
        self._updateNextStack()
        ExtruderManager.getInstance().resetSelectedObjectExtruders()
        self.activeExtruderChanged.emit()

    def getStack(self):
        if False:
            for i in range(10):
                print('nop')
        return self._stack