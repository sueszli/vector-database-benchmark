from UM.Job import Job
from UM.JobQueue import JobQueue
from UM.Logger import Logger
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Signal import Signal
import cura.CuraApplication
from cura.Machines.MachineNode import MachineNode
from cura.Settings.GlobalStack import GlobalStack
from typing import Dict, List, Optional, TYPE_CHECKING
import time
if TYPE_CHECKING:
    from cura.Machines.QualityGroup import QualityGroup
    from cura.Machines.QualityChangesGroup import QualityChangesGroup
    from UM.Settings.ContainerStack import ContainerStack

class ContainerTree:
    """This class contains a look-up tree for which containers are available at which stages of configuration.

    The tree starts at the machine definitions. For every distinct definition there will be one machine node here.

    All of the fallbacks for material choices, quality choices, etc. should be encoded in this tree. There must
    always be at least one child node (for nodes that have children) but that child node may be a node representing
    the empty instance container.
    """
    __instance = None

    @classmethod
    def getInstance(cls):
        if False:
            print('Hello World!')
        if cls.__instance is None:
            cls.__instance = ContainerTree()
        return cls.__instance

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.machines = self._MachineNodeMap()
        self.materialsChanged = Signal()
        cura.CuraApplication.CuraApplication.getInstance().initializationFinished.connect(self._onStartupFinished)

    def getCurrentQualityGroups(self) -> Dict[str, 'QualityGroup']:
        if False:
            for i in range(10):
                print('nop')
        'Get the quality groups available for the currently activated printer.\n\n        This contains all quality groups, enabled or disabled. To check whether the quality group can be activated,\n        test for the ``QualityGroup.is_available`` property.\n\n        :return: For every quality type, one quality group.\n        '
        global_stack = cura.CuraApplication.CuraApplication.getInstance().getGlobalContainerStack()
        if global_stack is None:
            return {}
        variant_names = [extruder.variant.getName() for extruder in global_stack.extruderList]
        material_bases = [extruder.material.getMetaDataEntry('base_file') for extruder in global_stack.extruderList]
        extruder_enabled = [extruder.isEnabled for extruder in global_stack.extruderList]
        return self.machines[global_stack.definition.getId()].getQualityGroups(variant_names, material_bases, extruder_enabled)

    def getCurrentQualityChangesGroups(self) -> List['QualityChangesGroup']:
        if False:
            print('Hello World!')
        'Get the quality changes groups available for the currently activated printer.\n\n        This contains all quality changes groups, enabled or disabled. To check whether the quality changes group can\n        be activated, test for the ``QualityChangesGroup.is_available`` property.\n\n        :return: A list of all quality changes groups.\n        '
        global_stack = cura.CuraApplication.CuraApplication.getInstance().getGlobalContainerStack()
        if global_stack is None:
            return []
        variant_names = [extruder.variant.getName() for extruder in global_stack.extruderList]
        material_bases = [extruder.material.getMetaDataEntry('base_file') for extruder in global_stack.extruderList]
        extruder_enabled = [extruder.isEnabled for extruder in global_stack.extruderList]
        return self.machines[global_stack.definition.getId()].getQualityChangesGroups(variant_names, material_bases, extruder_enabled)

    def _onStartupFinished(self) -> None:
        if False:
            while True:
                i = 10
        'Ran after completely starting up the application.'
        currently_added = ContainerRegistry.getInstance().findContainerStacks()
        JobQueue.getInstance().add(self._MachineNodeLoadJob(self, currently_added))

    class _MachineNodeMap:
        """Dictionary-like object that contains the machines.

        This handles the lazy loading of MachineNodes.
        """

        def __init__(self) -> None:
            if False:
                i = 10
                return i + 15
            self._machines = {}

        def __contains__(self, definition_id: str) -> bool:
            if False:
                i = 10
                return i + 15
            'Returns whether a printer with a certain definition ID exists.\n\n            This is regardless of whether or not the printer is loaded yet.\n\n            :param definition_id: The definition to look for.\n\n            :return: Whether or not a printer definition exists with that name.\n            '
            return len(ContainerRegistry.getInstance().findContainersMetadata(id=definition_id)) > 0

        def __getitem__(self, definition_id: str) -> MachineNode:
            if False:
                while True:
                    i = 10
            "Returns a machine node for the specified definition ID.\n\n            If the machine node wasn't loaded yet, this will load it lazily.\n\n            :param definition_id: The definition to look for.\n\n            :return: A machine node for that definition.\n            "
            if definition_id not in self._machines:
                start_time = time.time()
                self._machines[definition_id] = MachineNode(definition_id)
                self._machines[definition_id].materialsChanged.connect(ContainerTree.getInstance().materialsChanged)
                Logger.log('d', 'Adding container tree for {definition_id} took {duration} seconds.'.format(definition_id=definition_id, duration=time.time() - start_time))
            return self._machines[definition_id]

        def get(self, definition_id: str, default: Optional[MachineNode]=None) -> Optional[MachineNode]:
            if False:
                return 10
            "Gets a machine node for the specified definition ID, with default.\n\n            The default is returned if there is no definition with the specified ID. If the machine node wasn't\n            loaded yet, this will load it lazily.\n\n            :param definition_id: The definition to look for.\n            :param default: The machine node to return if there is no machine with that definition (can be ``None``\n            optionally or if not provided).\n\n            :return: A machine node for that definition, or the default if there is no definition with the provided\n            definition_id.\n            "
            if definition_id not in self:
                return default
            return self[definition_id]

        def is_loaded(self, definition_id: str) -> bool:
            if False:
                while True:
                    i = 10
            "Returns whether we've already cached this definition's node.\n\n            :param definition_id: The definition that we may have cached.\n\n            :return: ``True`` if it's cached.\n            "
            return definition_id in self._machines

    class _MachineNodeLoadJob(Job):
        """Pre-loads all currently added printers as a background task so that switching printers in the interface is
        faster.
        """

        def __init__(self, tree_root: 'ContainerTree', container_stacks: List['ContainerStack']) -> None:
            if False:
                return 10
            'Creates a new background task.\n\n            :param tree_root: The container tree instance. This cannot be obtained through the singleton static\n            function since the instance may not yet be constructed completely.\n            :param container_stacks: All of the stacks to pre-load the container trees for. This needs to be provided\n            from here because the stacks need to be constructed on the main thread because they are QObject.\n            '
            self.tree_root = tree_root
            self.container_stacks = container_stacks
            super().__init__()

        def run(self) -> None:
            if False:
                print('Hello World!')
            'Starts the background task.\n\n            The ``JobQueue`` will schedule this on a different thread.\n            '
            Logger.log('d', 'Started background loading of MachineNodes')
            for stack in self.container_stacks:
                if not isinstance(stack, GlobalStack):
                    continue
                time.sleep(0.5)
                definition_id = stack.definition.getId()
                if not self.tree_root.machines.is_loaded(definition_id):
                    _ = self.tree_root.machines[definition_id]
            Logger.log('d', 'All MachineNode loading completed')