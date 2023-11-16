from typing import TYPE_CHECKING
from UM.Logger import Logger
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Settings.Interfaces import ContainerInterface
from UM.Signal import Signal
from cura.Machines.ContainerNode import ContainerNode
from cura.Machines.MaterialNode import MaterialNode
import UM.FlameProfiler
if TYPE_CHECKING:
    from typing import Dict
    from cura.Machines.MachineNode import MachineNode

class VariantNode(ContainerNode):
    """This class represents an extruder variant in the container tree.

    The subnodes of these nodes are materials.

    This node contains materials with ALL filament diameters underneath it. The tree of this variant is not specific
    to one global stack, so because the list of materials can be different per stack depending on the compatible
    material diameter setting, we cannot filter them here. Filtering must be done in the model.
    """

    def __init__(self, container_id: str, machine: 'MachineNode') -> None:
        if False:
            return 10
        super().__init__(container_id)
        self.machine = machine
        self.materials = {}
        self.materialsChanged = Signal()
        container_registry = ContainerRegistry.getInstance()
        self.variant_name = container_registry.findContainersMetadata(id=container_id)[0]['name']
        container_registry.containerAdded.connect(self._materialAdded)
        container_registry.containerRemoved.connect(self._materialRemoved)
        self._loadAll()

    @UM.FlameProfiler.profile
    def _loadAll(self) -> None:
        if False:
            while True:
                i = 10
        '(Re)loads all materials under this variant.'
        container_registry = ContainerRegistry.getInstance()
        if not self.machine.has_materials:
            self.materials['empty_material'] = MaterialNode('empty_material', variant=self)
            return
        else:
            base_materials = container_registry.findInstanceContainersMetadata(type='material', definition='fdmprinter')
            printer_specific_materials = container_registry.findInstanceContainersMetadata(type='material', definition=self.machine.container_id)
            variant_specific_materials = container_registry.findInstanceContainersMetadata(type='material', definition=self.machine.container_id, variant_name=self.variant_name)
            materials_per_base_file = {material['base_file']: material for material in base_materials}
            materials_per_base_file.update({material['base_file']: material for material in printer_specific_materials})
            materials_per_base_file.update({material['base_file']: material for material in variant_specific_materials})
            materials = list(materials_per_base_file.values())
        filtered_materials = [material for material in materials if not self.machine.isExcludedMaterial(material)]
        for material in filtered_materials:
            base_file = material['base_file']
            if base_file not in self.materials:
                self.materials[base_file] = MaterialNode(material['id'], variant=self)
                self.materials[base_file].materialChanged.connect(self.materialsChanged)
        if not self.materials:
            self.materials['empty_material'] = MaterialNode('empty_material', variant=self)

    def preferredMaterial(self, approximate_diameter: int) -> MaterialNode:
        if False:
            for i in range(10):
                print('nop')
        'Finds the preferred material for this printer with this nozzle in one of the extruders.\n\n        If the preferred material is not available, an arbitrary material is returned. If there is a configuration\n        mistake (like a typo in the preferred material) this returns a random available material. If there are no\n        available materials, this will return the empty material node.\n\n        :param approximate_diameter: The desired approximate diameter of the material.\n\n        :return: The node for the preferred material, or any arbitrary material if there is no match.\n        '
        for (base_material, material_node) in self.materials.items():
            if self.machine.preferred_material == base_material and approximate_diameter == int(material_node.getMetaDataEntry('approximate_diameter')):
                return material_node
        if approximate_diameter == 2:
            preferred_material = self.machine.preferred_material + '_175'
            for (base_material, material_node) in self.materials.items():
                if preferred_material == base_material and approximate_diameter == int(material_node.getMetaDataEntry('approximate_diameter')):
                    return material_node
        for material_node in self.materials.values():
            if material_node.getMetaDataEntry('approximate_diameter') and approximate_diameter == int(material_node.getMetaDataEntry('approximate_diameter')):
                Logger.log('w', 'Could not find preferred material %s, falling back to whatever works', self.machine.preferred_material)
                return material_node
        fallback = next(iter(self.materials.values()))
        Logger.log('w', 'Could not find preferred material {preferred_material} with diameter {diameter} for variant {variant_id}, falling back to {fallback}.'.format(preferred_material=self.machine.preferred_material, diameter=approximate_diameter, variant_id=self.container_id, fallback=fallback.container_id))
        return fallback

    @UM.FlameProfiler.profile
    def _materialAdded(self, container: ContainerInterface) -> None:
        if False:
            i = 10
            return i + 15
        'When a material gets added to the set of profiles, we need to update our tree here.'
        if container.getMetaDataEntry('type') != 'material':
            return
        if not ContainerRegistry.getInstance().findContainersMetadata(id=container.getId()):
            Logger.log('d', 'Got container added signal for container [%s] but it no longer exists, do nothing.', container.getId())
            return
        if not self.machine.has_materials:
            return
        material_definition = container.getMetaDataEntry('definition')
        base_file = container.getMetaDataEntry('base_file')
        if base_file in self.machine.exclude_materials:
            return
        if base_file not in self.materials:
            if material_definition != 'fdmprinter' and material_definition != self.machine.container_id:
                return
            material_variant = container.getMetaDataEntry('variant_name')
            if material_variant is not None and material_variant != self.variant_name:
                return
        else:
            new_definition = container.getMetaDataEntry('definition')
            if new_definition == 'fdmprinter':
                return
            material_variant = container.getMetaDataEntry('variant_name')
            if new_definition != self.machine.container_id or material_variant != self.variant_name:
                return
            original_metadata = ContainerRegistry.getInstance().findContainersMetadata(id=self.materials[base_file].container_id)[0]
            if 'variant_name' in original_metadata or material_variant is None:
                return
        if 'empty_material' in self.materials:
            del self.materials['empty_material']
        self.materials[base_file] = MaterialNode(container.getId(), variant=self)
        self.materials[base_file].materialChanged.connect(self.materialsChanged)
        self.materialsChanged.emit(self.materials[base_file])

    @UM.FlameProfiler.profile
    def _materialRemoved(self, container: ContainerInterface) -> None:
        if False:
            for i in range(10):
                print('nop')
        if container.getMetaDataEntry('type') != 'material':
            return
        base_file = container.getMetaDataEntry('base_file')
        if base_file not in self.materials:
            return
        original_node = self.materials[base_file]
        del self.materials[base_file]
        self.materialsChanged.emit(original_node)
        materials_same_base_file = ContainerRegistry.getInstance().findContainersMetadata(base_file=base_file)
        if materials_same_base_file:
            most_specific_submaterial = None
            for submaterial in materials_same_base_file:
                if submaterial['definition'] == self.machine.container_id:
                    if submaterial.get('variant_name', 'empty') == self.variant_name:
                        most_specific_submaterial = submaterial
                        break
                    if submaterial.get('variant_name', 'empty') == 'empty':
                        most_specific_submaterial = submaterial
            if most_specific_submaterial is None:
                Logger.log('w', 'Material %s removed, but no suitable replacement found', base_file)
            else:
                Logger.log('i', 'Material %s (%s) overridden by %s', base_file, self.variant_name, most_specific_submaterial.get('id'))
                self.materials[base_file] = MaterialNode(most_specific_submaterial['id'], variant=self)
                self.materialsChanged.emit(self.materials[base_file])
        if not self.materials:
            self.materials['empty_material'] = MaterialNode('empty_material', variant=self)
            self.materialsChanged.emit(self.materials['empty_material'])