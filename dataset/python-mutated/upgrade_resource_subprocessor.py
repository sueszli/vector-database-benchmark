"""
Creates upgrade patches for resource modification effects in AoC.
"""
from __future__ import annotations
import typing
from .....nyan.nyan_structs import MemberOperator
from ....entity_object.conversion.aoc.genie_tech import GenieTechEffectBundleGroup
from ....entity_object.conversion.converter_object import RawAPIObject
from ....service.conversion import internal_name_lookups
from ....value_object.conversion.forward_ref import ForwardRef
if typing.TYPE_CHECKING:
    from openage.convert.entity_object.conversion.converter_object import ConverterObjectGroup
    from openage.nyan.nyan_structs import MemberOperator

class AoCUpgradeResourceSubprocessor:
    """
    Creates raw API objects for resource upgrade effects in AoC.
    """

    @staticmethod
    def berserk_heal_rate_upgrade(converter_group: ConverterObjectGroup, value: typing.Union[int, float], operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            i = 10
            return i + 15
        '\n        Creates a patch for the berserk heal rate modify effect (ID: 96).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: int, float\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        berserk_id = 692
        dataset = converter_group.data
        line = dataset.unit_lines[berserk_id]
        patches = []
        name_lookup_dict = internal_name_lookups.get_entity_lookups(dataset.game_version)
        obj_id = converter_group.get_id()
        if isinstance(converter_group, GenieTechEffectBundleGroup):
            tech_lookup_dict = internal_name_lookups.get_tech_lookups(dataset.game_version)
            obj_name = tech_lookup_dict[obj_id][0]
        else:
            civ_lookup_dict = internal_name_lookups.get_civ_lookups(dataset.game_version)
            obj_name = civ_lookup_dict[obj_id][0]
        game_entity_name = name_lookup_dict[berserk_id][0]
        patch_target_ref = f'{game_entity_name}.RegenerateHealth.HealthRate'
        patch_target_forward_ref = ForwardRef(line, patch_target_ref)
        wrapper_name = f'Change{game_entity_name}HealthRegenerationWrapper'
        wrapper_ref = f'{obj_name}.{wrapper_name}'
        wrapper_location = ForwardRef(converter_group, obj_name)
        wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects, wrapper_location)
        wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
        nyan_patch_name = f'Change{game_entity_name}HealthRegeneration'
        nyan_patch_ref = f'{obj_name}.{wrapper_name}.{nyan_patch_name}'
        nyan_patch_location = ForwardRef(converter_group, wrapper_ref)
        nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
        nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
        nyan_patch_raw_api_object.set_patch_target(patch_target_forward_ref)
        value = 1 / value
        nyan_patch_raw_api_object.add_raw_patch_member('rate', value, 'engine.util.attribute.AttributeRate', operator)
        patch_forward_ref = ForwardRef(converter_group, nyan_patch_ref)
        wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
        if team:
            team_property = dataset.pregen_nyan_objects['util.patch.property.types.Team'].get_nyan_object()
            properties = {dataset.nyan_api_objects['engine.util.patch.property.type.Diplomatic']: team_property}
            wrapper_raw_api_object.add_raw_member('properties', properties, 'engine.util.patch.Patch')
        converter_group.add_raw_api_object(wrapper_raw_api_object)
        converter_group.add_raw_api_object(nyan_patch_raw_api_object)
        wrapper_forward_ref = ForwardRef(converter_group, wrapper_ref)
        patches.append(wrapper_forward_ref)
        return patches

    @staticmethod
    def bonus_population_upgrade(converter_group: ConverterObjectGroup, value: typing.Union[int, float], operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            while True:
                i = 10
        '\n        Creates a patch for the bonus population effect (ID: 32).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: int, float\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        dataset = converter_group.data
        patches = []
        obj_id = converter_group.get_id()
        if isinstance(converter_group, GenieTechEffectBundleGroup):
            tech_lookup_dict = internal_name_lookups.get_tech_lookups(dataset.game_version)
            obj_name = tech_lookup_dict[obj_id][0]
        else:
            civ_lookup_dict = internal_name_lookups.get_civ_lookups(dataset.game_version)
            obj_name = civ_lookup_dict[obj_id][0]
        patch_target_ref = 'util.resource.types.PopulationSpace'
        patch_target = dataset.pregen_nyan_objects[patch_target_ref].get_nyan_object()
        wrapper_name = 'ChangePopulationCapWrapper'
        wrapper_ref = f'{obj_name}.{wrapper_name}'
        wrapper_location = ForwardRef(converter_group, obj_name)
        wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects, wrapper_location)
        wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
        nyan_patch_name = 'ChangePopulationCap'
        nyan_patch_ref = f'{obj_name}.{wrapper_name}.{nyan_patch_name}'
        nyan_patch_location = ForwardRef(converter_group, wrapper_ref)
        nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
        nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
        nyan_patch_raw_api_object.set_patch_target(patch_target)
        nyan_patch_raw_api_object.add_raw_patch_member('max_amount', value, 'engine.util.resource.ResourceContingent', operator)
        patch_forward_ref = ForwardRef(converter_group, nyan_patch_ref)
        wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
        if team:
            team_property = dataset.pregen_nyan_objects['util.patch.property.types.Team'].get_nyan_object()
            properties = {dataset.nyan_api_objects['engine.util.patch.property.type.Diplomatic']: team_property}
            wrapper_raw_api_object.add_raw_member('properties', properties, 'engine.util.patch.Patch')
        converter_group.add_raw_api_object(wrapper_raw_api_object)
        converter_group.add_raw_api_object(nyan_patch_raw_api_object)
        wrapper_forward_ref = ForwardRef(converter_group, wrapper_ref)
        patches.append(wrapper_forward_ref)
        return patches

    @staticmethod
    def building_conversion_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            return 10
        '\n        Creates a patch for the building conversion effect (ID: 28).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        monk_id = 125
        dataset = converter_group.data
        line = dataset.unit_lines[monk_id]
        patches = []
        name_lookup_dict = internal_name_lookups.get_entity_lookups(dataset.game_version)
        obj_id = converter_group.get_id()
        if isinstance(converter_group, GenieTechEffectBundleGroup):
            tech_lookup_dict = internal_name_lookups.get_tech_lookups(dataset.game_version)
            obj_name = tech_lookup_dict[obj_id][0]
        else:
            civ_lookup_dict = internal_name_lookups.get_civ_lookups(dataset.game_version)
            obj_name = civ_lookup_dict[obj_id][0]
        game_entity_name = name_lookup_dict[monk_id][0]
        patch_target_ref = f'{game_entity_name}.Convert'
        patch_target_forward_ref = ForwardRef(line, patch_target_ref)
        wrapper_name = 'EnableBuildingConversionWrapper'
        wrapper_ref = f'{obj_name}.{wrapper_name}'
        wrapper_location = ForwardRef(converter_group, obj_name)
        wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects, wrapper_location)
        wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
        nyan_patch_name = 'EnableBuildingConversion'
        nyan_patch_ref = f'{obj_name}.{wrapper_name}.{nyan_patch_name}'
        nyan_patch_location = ForwardRef(converter_group, wrapper_ref)
        nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
        nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
        nyan_patch_raw_api_object.set_patch_target(patch_target_forward_ref)
        allowed_types = [dataset.pregen_nyan_objects['util.game_entity_type.types.Building'].get_nyan_object()]
        nyan_patch_raw_api_object.add_raw_patch_member('allowed_types', allowed_types, 'engine.ability.type.ApplyDiscreteEffect', MemberOperator.ADD)
        tc_line = dataset.building_lines[109]
        farm_line = dataset.building_lines[50]
        fish_trap_line = dataset.building_lines[199]
        monastery_line = dataset.building_lines[104]
        castle_line = dataset.building_lines[82]
        palisade_line = dataset.building_lines[72]
        stone_wall_line = dataset.building_lines[117]
        stone_gate_line = dataset.building_lines[64]
        wonder_line = dataset.building_lines[276]
        blacklisted_forward_refs = [ForwardRef(tc_line, 'TownCenter'), ForwardRef(farm_line, 'Farm'), ForwardRef(fish_trap_line, 'FishingTrap'), ForwardRef(monastery_line, 'Monastery'), ForwardRef(castle_line, 'Castle'), ForwardRef(palisade_line, 'PalisadeWall'), ForwardRef(stone_wall_line, 'StoneWall'), ForwardRef(stone_gate_line, 'StoneGate'), ForwardRef(wonder_line, 'Wonder')]
        nyan_patch_raw_api_object.add_raw_patch_member('blacklisted_entities', blacklisted_forward_refs, 'engine.ability.type.ApplyDiscreteEffect', MemberOperator.ADD)
        patch_forward_ref = ForwardRef(converter_group, nyan_patch_ref)
        wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
        converter_group.add_raw_api_object(wrapper_raw_api_object)
        converter_group.add_raw_api_object(nyan_patch_raw_api_object)
        wrapper_forward_ref = ForwardRef(converter_group, wrapper_ref)
        patches.append(wrapper_forward_ref)
        wrapper_name = 'EnableSiegeUnitConversionWrapper'
        wrapper_ref = f'{obj_name}.{wrapper_name}'
        wrapper_location = ForwardRef(converter_group, obj_name)
        wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects, wrapper_location)
        wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
        nyan_patch_name = 'EnableSiegeUnitConversion'
        nyan_patch_ref = f'{obj_name}.{wrapper_name}.{nyan_patch_name}'
        nyan_patch_location = ForwardRef(converter_group, wrapper_ref)
        nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
        nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
        nyan_patch_raw_api_object.set_patch_target(patch_target_forward_ref)
        blacklisted_entities = []
        for unit_line in dataset.unit_lines.values():
            if unit_line.get_class_id() in (13, 55):
                blacklisted_name = name_lookup_dict[unit_line.get_head_unit_id()][0]
                blacklisted_entities.append(ForwardRef(unit_line, blacklisted_name))
        nyan_patch_raw_api_object.add_raw_patch_member('blacklisted_entities', blacklisted_entities, 'engine.ability.type.ApplyDiscreteEffect', MemberOperator.SUBTRACT)
        patch_forward_ref = ForwardRef(converter_group, nyan_patch_ref)
        wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
        if team:
            team_property = dataset.pregen_nyan_objects['util.patch.property.types.Team'].get_nyan_object()
            properties = {dataset.nyan_api_objects['engine.util.patch.property.type.Diplomatic']: team_property}
            wrapper_raw_api_object.add_raw_member('properties', properties, 'engine.util.patch.Patch')
        converter_group.add_raw_api_object(wrapper_raw_api_object)
        converter_group.add_raw_api_object(nyan_patch_raw_api_object)
        wrapper_forward_ref = ForwardRef(converter_group, wrapper_ref)
        patches.append(wrapper_forward_ref)
        return patches

    @staticmethod
    def chinese_tech_discount_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            while True:
                i = 10
        '\n        Creates a patch for the chinese tech discount effect (ID: 85).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def construction_speed_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            print('Hello World!')
        '\n        Creates a patch for the construction speed modify effect (ID: 195).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def conversion_resistance_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a patch for the conversion resistance modify effect (ID: 77).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def conversion_resistance_min_rounds_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            i = 10
            return i + 15
        '\n        Creates a patch for the conversion resistance modify effect (ID: 178).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def conversion_resistance_max_rounds_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            i = 10
            return i + 15
        '\n        Creates a patch for the conversion resistance modify effect (ID: 179).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def crenellations_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            i = 10
            return i + 15
        '\n        Creates a patch for the crenellations effect (ID: 194).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def faith_recharge_rate_upgrade(converter_group: ConverterObjectGroup, value: typing.Union[int, float], operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            print('Hello World!')
        '\n        Creates a patch for the faith_recharge_rate modify effect (ID: 35).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: int, float\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        monk_id = 125
        dataset = converter_group.data
        line = dataset.unit_lines[monk_id]
        patches = []
        name_lookup_dict = internal_name_lookups.get_entity_lookups(dataset.game_version)
        obj_id = converter_group.get_id()
        if isinstance(converter_group, GenieTechEffectBundleGroup):
            tech_lookup_dict = internal_name_lookups.get_tech_lookups(dataset.game_version)
            obj_name = tech_lookup_dict[obj_id][0]
        else:
            civ_lookup_dict = internal_name_lookups.get_civ_lookups(dataset.game_version)
            obj_name = civ_lookup_dict[obj_id][0]
        game_entity_name = name_lookup_dict[monk_id][0]
        patch_target_ref = f'{game_entity_name}.RegenerateFaith.FaithRate'
        patch_target_forward_ref = ForwardRef(line, patch_target_ref)
        wrapper_name = f'Change{game_entity_name}FaithRegenerationWrapper'
        wrapper_ref = f'{obj_name}.{wrapper_name}'
        wrapper_location = ForwardRef(converter_group, obj_name)
        wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects, wrapper_location)
        wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
        nyan_patch_name = f'Change{game_entity_name}FaithRegeneration'
        nyan_patch_ref = f'{obj_name}.{wrapper_name}.{nyan_patch_name}'
        nyan_patch_location = ForwardRef(converter_group, wrapper_ref)
        nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
        nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
        nyan_patch_raw_api_object.set_patch_target(patch_target_forward_ref)
        nyan_patch_raw_api_object.add_raw_patch_member('rate', value, 'engine.util.attribute.AttributeRate', operator)
        patch_forward_ref = ForwardRef(converter_group, nyan_patch_ref)
        wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
        if team:
            team_property = dataset.pregen_nyan_objects['util.patch.property.types.Team'].get_nyan_object()
            properties = {dataset.nyan_api_objects['engine.util.patch.property.type.Diplomatic']: team_property}
            wrapper_raw_api_object.add_raw_member('properties', properties, 'engine.util.patch.Patch')
        converter_group.add_raw_api_object(wrapper_raw_api_object)
        converter_group.add_raw_api_object(nyan_patch_raw_api_object)
        wrapper_forward_ref = ForwardRef(converter_group, wrapper_ref)
        patches.append(wrapper_forward_ref)
        return patches

    @staticmethod
    def farm_food_upgrade(converter_group: ConverterObjectGroup, value: typing.Union[int, float], operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a patch for the farm food modify effect (ID: 36).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: int, float\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        farm_id = 50
        dataset = converter_group.data
        line = dataset.building_lines[farm_id]
        patches = []
        name_lookup_dict = internal_name_lookups.get_entity_lookups(dataset.game_version)
        obj_id = converter_group.get_id()
        if isinstance(converter_group, GenieTechEffectBundleGroup):
            tech_lookup_dict = internal_name_lookups.get_tech_lookups(dataset.game_version)
            obj_name = tech_lookup_dict[obj_id][0]
        else:
            civ_lookup_dict = internal_name_lookups.get_civ_lookups(dataset.game_version)
            obj_name = civ_lookup_dict[obj_id][0]
        game_entity_name = name_lookup_dict[farm_id][0]
        patch_target_ref = f'{game_entity_name}.Harvestable.{game_entity_name}ResourceSpot'
        patch_target_forward_ref = ForwardRef(line, patch_target_ref)
        wrapper_name = f'Change{game_entity_name}FoodAmountWrapper'
        wrapper_ref = f'{obj_name}.{wrapper_name}'
        wrapper_location = ForwardRef(converter_group, obj_name)
        wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects, wrapper_location)
        wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
        nyan_patch_name = f'Change{game_entity_name}FoodAmount'
        nyan_patch_ref = f'{obj_name}.{wrapper_name}.{nyan_patch_name}'
        nyan_patch_location = ForwardRef(converter_group, wrapper_ref)
        nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
        nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
        nyan_patch_raw_api_object.set_patch_target(patch_target_forward_ref)
        nyan_patch_raw_api_object.add_raw_patch_member('max_amount', value, 'engine.util.resource_spot.ResourceSpot', operator)
        patch_forward_ref = ForwardRef(converter_group, nyan_patch_ref)
        wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
        if team:
            team_property = dataset.pregen_nyan_objects['util.patch.property.types.Team'].get_nyan_object()
            properties = {dataset.nyan_api_objects['engine.util.patch.property.type.Diplomatic']: team_property}
            wrapper_raw_api_object.add_raw_member('properties', properties, 'engine.util.patch.Patch')
        converter_group.add_raw_api_object(wrapper_raw_api_object)
        converter_group.add_raw_api_object(nyan_patch_raw_api_object)
        wrapper_forward_ref = ForwardRef(converter_group, wrapper_ref)
        patches.append(wrapper_forward_ref)
        return patches

    @staticmethod
    def gather_food_efficiency_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            while True:
                i = 10
        '\n        Creates a patch for the food gathering efficiency modify effect (ID: 190).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def gather_wood_efficiency_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            i = 10
            return i + 15
        '\n        Creates a patch for the wood gathering efficiency modify effect (ID: 189).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def gather_gold_efficiency_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            print('Hello World!')
        '\n        Creates a patch for the gold gathering efficiency modify effect (ID: 47).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def gather_stone_efficiency_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            return 10
        '\n        Creates a patch for the stone gathering efficiency modify effect (ID: 79).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def heal_range_upgrade(converter_group: ConverterObjectGroup, value: typing.Union[int, float], operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            while True:
                i = 10
        '\n        Creates a patch for the heal range modify effect (ID: 90).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: int, float\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        monk_id = 125
        dataset = converter_group.data
        line = dataset.unit_lines[monk_id]
        patches = []
        name_lookup_dict = internal_name_lookups.get_entity_lookups(dataset.game_version)
        obj_id = converter_group.get_id()
        if isinstance(converter_group, GenieTechEffectBundleGroup):
            tech_lookup_dict = internal_name_lookups.get_tech_lookups(dataset.game_version)
            obj_name = tech_lookup_dict[obj_id][0]
        else:
            civ_lookup_dict = internal_name_lookups.get_civ_lookups(dataset.game_version)
            obj_name = civ_lookup_dict[obj_id][0]
        game_entity_name = name_lookup_dict[monk_id][0]
        patch_target_ref = f'{game_entity_name}.Heal'
        patch_target_forward_ref = ForwardRef(line, patch_target_ref)
        wrapper_name = f'Change{game_entity_name}HealRangeWrapper'
        wrapper_ref = f'{obj_name}.{wrapper_name}'
        wrapper_location = ForwardRef(converter_group, obj_name)
        wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects, wrapper_location)
        wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
        nyan_patch_name = f'Change{game_entity_name}HealRange'
        nyan_patch_ref = f'{obj_name}.{wrapper_name}.{nyan_patch_name}'
        nyan_patch_location = ForwardRef(converter_group, wrapper_ref)
        nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
        nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
        nyan_patch_raw_api_object.set_patch_target(patch_target_forward_ref)
        nyan_patch_raw_api_object.add_raw_patch_member('max_range', value, 'engine.ability.type.RangedContinuousEffect', operator)
        patch_forward_ref = ForwardRef(converter_group, nyan_patch_ref)
        wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
        if team:
            team_property = dataset.pregen_nyan_objects['util.patch.property.types.Team'].get_nyan_object()
            properties = {dataset.nyan_api_objects['engine.util.patch.property.type.Diplomatic']: team_property}
            wrapper_raw_api_object.add_raw_member('properties', properties, 'engine.util.patch.Patch')
        converter_group.add_raw_api_object(wrapper_raw_api_object)
        converter_group.add_raw_api_object(nyan_patch_raw_api_object)
        wrapper_forward_ref = ForwardRef(converter_group, wrapper_ref)
        patches.append(wrapper_forward_ref)
        return patches

    @staticmethod
    def heal_rate_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a patch for the heal rate modify effect (ID: 89).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def herding_dominance_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            return 10
        '\n        Creates a patch for the herding dominance effect (ID: 97).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def heresy_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            while True:
                i = 10
        '\n        Creates a patch for the heresy effect (ID: 192).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def monk_conversion_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            return 10
        '\n        Creates a patch for the monk conversion effect (ID: 27).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        monk_id = 125
        dataset = converter_group.data
        line = dataset.unit_lines[monk_id]
        patches = []
        name_lookup_dict = internal_name_lookups.get_entity_lookups(dataset.game_version)
        obj_id = converter_group.get_id()
        if isinstance(converter_group, GenieTechEffectBundleGroup):
            tech_lookup_dict = internal_name_lookups.get_tech_lookups(dataset.game_version)
            obj_name = tech_lookup_dict[obj_id][0]
        else:
            civ_lookup_dict = internal_name_lookups.get_civ_lookups(dataset.game_version)
            obj_name = civ_lookup_dict[obj_id][0]
        game_entity_name = name_lookup_dict[monk_id][0]
        patch_target_ref = f'{game_entity_name}.Convert'
        patch_target_forward_ref = ForwardRef(line, patch_target_ref)
        wrapper_name = f'Enable{game_entity_name}ConversionWrapper'
        wrapper_ref = f'{obj_name}.{wrapper_name}'
        wrapper_location = ForwardRef(converter_group, obj_name)
        wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects, wrapper_location)
        wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
        nyan_patch_name = f'Enable{game_entity_name}Conversion'
        nyan_patch_ref = f'{obj_name}.{wrapper_name}.{nyan_patch_name}'
        nyan_patch_location = ForwardRef(converter_group, wrapper_ref)
        nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
        nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
        nyan_patch_raw_api_object.set_patch_target(patch_target_forward_ref)
        monk_forward_ref = ForwardRef(line, game_entity_name)
        nyan_patch_raw_api_object.add_raw_patch_member('blacklisted_entities', [monk_forward_ref], 'engine.ability.type.ApplyDiscreteEffect', MemberOperator.SUBTRACT)
        patch_forward_ref = ForwardRef(converter_group, nyan_patch_ref)
        wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
        if team:
            team_property = dataset.pregen_nyan_objects['util.patch.property.types.Team'].get_nyan_object()
            properties = {dataset.nyan_api_objects['engine.util.patch.property.type.Diplomatic']: team_property}
            wrapper_raw_api_object.add_raw_member('properties', properties, 'engine.util.patch.Patch')
        converter_group.add_raw_api_object(wrapper_raw_api_object)
        converter_group.add_raw_api_object(nyan_patch_raw_api_object)
        wrapper_forward_ref = ForwardRef(converter_group, wrapper_ref)
        patches.append(wrapper_forward_ref)
        return patches

    @staticmethod
    def relic_gold_bonus_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            print('Hello World!')
        '\n        Creates a patch for the relic gold bonus modify effect (ID: 191).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def research_time_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            return 10
        '\n        Creates a patch for the research time modify effect (ID: 86).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def reveal_ally_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a patch for the reveal ally modify effect (ID: 50).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def reveal_enemy_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            print('Hello World!')
        '\n        Creates a patch for the reveal enemy modify effect (ID: 183).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def siege_conversion_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            return 10
        '\n        Creates a patch for the siege conversion effect (ID: 29).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def ship_conversion_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            print('Hello World!')
        '\n        Creates a patch for the ship conversion effect (ID: 87).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def spies_discount_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            i = 10
            return i + 15
        '\n        Creates a patch for the spies discount effect (ID: 197).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def starting_food_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            while True:
                i = 10
        '\n        Creates a patch for the starting food modify effect (ID: 91).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: MemberOperator\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def starting_wood_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a patch for the starting wood modify effect (ID: 92).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def starting_stone_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            print('Hello World!')
        '\n        Creates a patch for the starting stone modify effect (ID: 93).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def starting_gold_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            print('Hello World!')
        '\n        Creates a patch for the starting gold modify effect (ID: 94).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def starting_villagers_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            return 10
        '\n        Creates a patch for the starting villagers modify effect (ID: 84).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def starting_population_space_upgrade(converter_group: ConverterObjectGroup, value: typing.Union[int, float], operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            i = 10
            return i + 15
        '\n        Creates a patch for the starting popspace modify effect (ID: 4).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: int, float\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        dataset = converter_group.data
        patches = []
        obj_id = converter_group.get_id()
        if isinstance(converter_group, GenieTechEffectBundleGroup):
            tech_lookup_dict = internal_name_lookups.get_tech_lookups(dataset.game_version)
            obj_name = tech_lookup_dict[obj_id][0]
        else:
            civ_lookup_dict = internal_name_lookups.get_civ_lookups(dataset.game_version)
            obj_name = civ_lookup_dict[obj_id][0]
        patch_target_ref = 'util.resource.types.PopulationSpace'
        patch_target = dataset.pregen_nyan_objects[patch_target_ref].get_nyan_object()
        wrapper_name = 'ChangeInitialPopulationLimitWrapper'
        wrapper_ref = f'{obj_name}.{wrapper_name}'
        wrapper_location = ForwardRef(converter_group, obj_name)
        wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects, wrapper_location)
        wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
        nyan_patch_name = 'ChangeInitialPopulationLimit'
        nyan_patch_ref = f'{obj_name}.{wrapper_name}.{nyan_patch_name}'
        nyan_patch_location = ForwardRef(converter_group, wrapper_ref)
        nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
        nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
        nyan_patch_raw_api_object.set_patch_target(patch_target)
        nyan_patch_raw_api_object.add_raw_patch_member('min_amount', value, 'engine.util.resource.ResourceContingent', operator)
        patch_forward_ref = ForwardRef(converter_group, nyan_patch_ref)
        wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
        if team:
            team_property = dataset.pregen_nyan_objects['util.patch.property.types.Team'].get_nyan_object()
            properties = {dataset.nyan_api_objects['engine.util.patch.property.type.Diplomatic']: team_property}
            wrapper_raw_api_object.add_raw_member('properties', properties, 'engine.util.patch.Patch')
        converter_group.add_raw_api_object(wrapper_raw_api_object)
        converter_group.add_raw_api_object(nyan_patch_raw_api_object)
        wrapper_forward_ref = ForwardRef(converter_group, wrapper_ref)
        patches.append(wrapper_forward_ref)
        return patches

    @staticmethod
    def theocracy_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            while True:
                i = 10
        '\n        Creates a patch for the theocracy effect (ID: 193).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def trade_penalty_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            while True:
                i = 10
        '\n        Creates a patch for the trade penalty modify effect (ID: 78).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def tribute_inefficiency_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            while True:
                i = 10
        '\n        Creates a patch for the tribute inefficiency modify effect (ID: 46).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches

    @staticmethod
    def wonder_time_increase_upgrade(converter_group: ConverterObjectGroup, value: typing.Any, operator: MemberOperator, team: bool=False) -> list[ForwardRef]:
        if False:
            print('Hello World!')
        '\n        Creates a patch for the wonder time modify effect (ID: 196).\n\n        :param converter_group: Tech/Civ that gets the patch.\n        :type converter_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param value: Value used for patching the member.\n        :type value: Any\n        :param operator: Operator used for patching the member.\n        :type operator: MemberOperator\n        :returns: The forward references for the generated patches.\n        :rtype: list\n        '
        patches = []
        return patches