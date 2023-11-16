from typing import Dict, List
from UM.Logger import Logger
from UM.Signal import Signal
from UM.Util import parseBool
from UM.Settings.ContainerRegistry import ContainerRegistry
import cura.CuraApplication
from cura.Machines.ContainerNode import ContainerNode
from cura.Machines.QualityChangesGroup import QualityChangesGroup
from cura.Machines.QualityGroup import QualityGroup
from cura.Machines.QualityNode import QualityNode
from cura.Machines.VariantNode import VariantNode
from cura.Machines.MaterialNode import MaterialNode
import UM.FlameProfiler

class MachineNode(ContainerNode):
    """This class represents a machine in the container tree.

    The subnodes of these nodes are variants.
    """

    def __init__(self, container_id: str) -> None:
        if False:
            print('Hello World!')
        super().__init__(container_id)
        self.variants = {}
        self.global_qualities = {}
        self.materialsChanged = Signal()
        container_registry = ContainerRegistry.getInstance()
        try:
            my_metadata = container_registry.findContainersMetadata(id=container_id)[0]
        except IndexError:
            Logger.log('Unable to find metadata for container %s', container_id)
            my_metadata = {}
        self.has_materials = parseBool(my_metadata.get('has_materials', 'true'))
        self.has_variants = parseBool(my_metadata.get('has_variants', 'false'))
        self.has_machine_quality = parseBool(my_metadata.get('has_machine_quality', 'false'))
        self.quality_definition = my_metadata.get('quality_definition', container_id) if self.has_machine_quality else 'fdmprinter'
        self.exclude_materials = my_metadata.get('exclude_materials', [])
        self.preferred_variant_name = my_metadata.get('preferred_variant_name', '')
        self.preferred_material = my_metadata.get('preferred_material', '')
        self.preferred_quality_type = my_metadata.get('preferred_quality_type', '')
        self._loadAll()

    def getQualityGroups(self, variant_names: List[str], material_bases: List[str], extruder_enabled: List[bool]) -> Dict[str, QualityGroup]:
        if False:
            i = 10
            return i + 15
        'Get the available quality groups for this machine.\n\n        This returns all quality groups, regardless of whether they are available to the combination of extruders or\n        not. On the resulting quality groups, the is_available property is set to indicate whether the quality group\n        can be selected according to the combination of extruders in the parameters.\n\n        :param variant_names: The names of the variants loaded in each extruder.\n        :param material_bases: The base file names of the materials loaded in each extruder.\n        :param extruder_enabled: Whether or not the extruders are enabled. This allows the function to set the\n        is_available properly.\n\n        :return: For each available quality type, a QualityGroup instance.\n        '
        if len(variant_names) != len(material_bases) or len(variant_names) != len(extruder_enabled):
            Logger.log('e', 'The number of extruders in the list of variants (' + str(len(variant_names)) + ') is not equal to the number of extruders in the list of materials (' + str(len(material_bases)) + ') or the list of enabled extruders (' + str(len(extruder_enabled)) + ').')
            return {}
        qualities_per_type_per_extruder = [{}] * len(variant_names)
        for (extruder_nr, variant_name) in enumerate(variant_names):
            if not extruder_enabled[extruder_nr]:
                continue
            material_base = material_bases[extruder_nr]
            if variant_name not in self.variants or material_base not in self.variants[variant_name].materials:
                qualities_per_type_per_extruder[extruder_nr] = self.global_qualities
            else:
                qualities_per_type_per_extruder[extruder_nr] = {node.quality_type: node for node in self.variants[variant_name].materials[material_base].qualities.values()}
        quality_groups = {}
        for (quality_type, global_quality_node) in self.global_qualities.items():
            if not global_quality_node.container:
                Logger.log('w', "Node {0} doesn't have a container.".format(global_quality_node.container_id))
                continue
            quality_groups[quality_type] = QualityGroup(name=global_quality_node.getMetaDataEntry('name', 'Unnamed profile'), quality_type=quality_type)
            quality_groups[quality_type].node_for_global = global_quality_node
            for (extruder_position, qualities_per_type) in enumerate(qualities_per_type_per_extruder):
                if quality_type in qualities_per_type:
                    quality_groups[quality_type].setExtruderNode(extruder_position, qualities_per_type[quality_type])
        available_quality_types = set(quality_groups.keys())
        for (extruder_nr, qualities_per_type) in enumerate(qualities_per_type_per_extruder):
            if not extruder_enabled[extruder_nr]:
                continue
            available_quality_types.intersection_update(qualities_per_type.keys())
        for quality_type in available_quality_types:
            quality_groups[quality_type].is_available = True
        return quality_groups

    def getQualityChangesGroups(self, variant_names: List[str], material_bases: List[str], extruder_enabled: List[bool]) -> List[QualityChangesGroup]:
        if False:
            while True:
                i = 10
        'Returns all of the quality changes groups available to this printer.\n\n        The quality changes groups store which quality type and intent category they were made for, but not which\n        material and nozzle. Instead for the quality type and intent category, the quality changes will always be\n        available but change the quality type and intent category when activated.\n\n        The quality changes group does depend on the printer: Which quality definition is used.\n\n        The quality changes groups that are available do depend on the quality types that are available, so it must\n        still be known which extruders are enabled and which materials and variants are loaded in them. This allows\n        setting the correct is_available flag.\n\n        :param variant_names: The names of the variants loaded in each extruder.\n        :param material_bases: The base file names of the materials loaded in each extruder.\n        :param extruder_enabled: For each extruder whether or not they are enabled.\n\n        :return: List of all quality changes groups for the printer.\n        '
        machine_quality_changes = ContainerRegistry.getInstance().findContainersMetadata(type='quality_changes', definition=self.quality_definition)
        groups_by_name = {}
        for quality_changes in machine_quality_changes:
            name = quality_changes['name']
            if name not in groups_by_name:
                from cura.CuraApplication import CuraApplication
                groups_by_name[name] = QualityChangesGroup(name, quality_type=quality_changes['quality_type'], intent_category=quality_changes.get('intent_category', 'default'), parent=CuraApplication.getInstance())
            elif groups_by_name[name].intent_category == 'default':
                groups_by_name[name].intent_category = quality_changes.get('intent_category', 'default')
            if quality_changes.get('position') is not None and quality_changes.get('position') != 'None':
                groups_by_name[name].metadata_per_extruder[int(quality_changes['position'])] = quality_changes
            else:
                groups_by_name[name].metadata_for_global = quality_changes
        quality_groups = self.getQualityGroups(variant_names, material_bases, extruder_enabled)
        for quality_changes_group in groups_by_name.values():
            if quality_changes_group.quality_type not in quality_groups:
                if quality_changes_group.quality_type == 'not_supported':
                    quality_changes_group.is_available = True
                else:
                    quality_changes_group.is_available = False
            else:
                quality_changes_group.is_available = quality_groups[quality_changes_group.quality_type].is_available
        return list(groups_by_name.values())

    def preferredGlobalQuality(self) -> 'QualityNode':
        if False:
            return 10
        'Gets the preferred global quality node, going by the preferred quality type.\n\n        If the preferred global quality is not in there, an arbitrary global quality is taken. If there are no global\n        qualities, an empty quality is returned.\n        '
        return self.global_qualities.get(self.preferred_quality_type, next(iter(self.global_qualities.values())))

    def isExcludedMaterial(self, material: MaterialNode) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Returns whether the material should be excluded from the list of materials.'
        for exclude_material in self.exclude_materials:
            if exclude_material in material['id']:
                return True
        return False

    @UM.FlameProfiler.profile
    def _loadAll(self) -> None:
        if False:
            i = 10
            return i + 15
        '(Re)loads all variants under this printer.'
        container_registry = ContainerRegistry.getInstance()
        if not self.has_variants:
            self.variants['empty'] = VariantNode('empty_variant', machine=self)
            self.variants['empty'].materialsChanged.connect(self.materialsChanged)
        else:
            variants = container_registry.findInstanceContainersMetadata(type='variant', definition=self.container_id, hardware_type='nozzle')
            for variant in variants:
                variant_name = variant['name']
                if variant_name not in self.variants:
                    self.variants[variant_name] = VariantNode(variant['id'], machine=self)
                    self.variants[variant_name].materialsChanged.connect(self.materialsChanged)
                else:
                    self.variants[variant_name]._loadAll()
            if not self.variants:
                self.variants['empty'] = VariantNode('empty_variant', machine=self)
        global_qualities = container_registry.findInstanceContainersMetadata(type='quality', definition=self.quality_definition, global_quality='True')
        if not global_qualities:
            global_qualities = container_registry.findInstanceContainersMetadata(type='quality', definition='fdmprinter', global_quality='True')
            if not global_qualities:
                global_qualities = [cura.CuraApplication.CuraApplication.getInstance().empty_quality_container.getMetaData()]
        for global_quality in global_qualities:
            self.global_qualities[global_quality['quality_type']] = QualityNode(global_quality['id'], parent=self)