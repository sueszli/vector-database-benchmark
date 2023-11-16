"""
Upgrades effects and resistances for the Apply*Effect and Resistance
abilities.
"""
from __future__ import annotations
import typing
from .....nyan.nyan_structs import MemberOperator
from ....entity_object.conversion.aoc.genie_unit import GenieBuildingLineGroup
from ....entity_object.conversion.converter_object import RawAPIObject
from ....service.conversion import internal_name_lookups
from ....value_object.conversion.forward_ref import ForwardRef
from ....value_object.read.value_members import NoDiffMember, LeftMissingMember, RightMissingMember
if typing.TYPE_CHECKING:
    from openage.convert.entity_object.conversion.converter_object import ConverterObject
    from openage.convert.entity_object.conversion.aoc.genie_tech import GenieTechEffectBundleGroup
    from openage.convert.entity_object.conversion.aoc.genie_unit import GenieGameEntityGroup

class AoCUpgradeEffectSubprocessor:
    """
    Creates raw API objects for attack/resistance upgrades in AoC.
    """

    @staticmethod
    def get_attack_effects(tech_group: GenieTechEffectBundleGroup, line: GenieGameEntityGroup, diff: ConverterObject, ability_ref: str) -> list[ForwardRef]:
        if False:
            return 10
        '\n        Upgrades effects that are used for attacking (unit command: 7)\n\n        :param tech_group: Tech that gets the patch.\n        :type tech_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param line: Unit/Building line that gets the ability.\n        :type line: ...dataformat.converter_object.ConverterObjectGroup\n        :param diff: A diff between two ConvertObject instances.\n        :type diff: ...dataformat.converter_object.ConverterObject\n        :param ability_ref: Reference of the ability raw API object the effects are added to.\n        :type ability_ref: str\n        :returns: The forward references for the effects.\n        :rtype: list\n        '
        head_unit_id = line.get_head_unit_id()
        tech_id = tech_group.get_id()
        dataset = line.data
        patches = []
        name_lookup_dict = internal_name_lookups.get_entity_lookups(dataset.game_version)
        armor_lookup_dict = internal_name_lookups.get_armor_class_lookups(dataset.game_version)
        tech_lookup_dict = internal_name_lookups.get_tech_lookups(dataset.game_version)
        tech_name = tech_lookup_dict[tech_id][0]
        diff_attacks = diff['attacks'].value
        for diff_attack in diff_attacks.values():
            if isinstance(diff_attack, NoDiffMember):
                continue
            if isinstance(diff_attack, LeftMissingMember):
                attack = diff_attack.ref
                armor_class = attack['type_id'].value
                attack_amount = attack['amount'].value
                if armor_class == -1:
                    continue
                class_name = armor_lookup_dict[armor_class]
                effect_parent = 'engine.effect.discrete.flat_attribute_change.FlatAttributeChange'
                attack_parent = 'engine.effect.discrete.flat_attribute_change.type.FlatAttributeChangeDecrease'
                patch_target_ref = f'{ability_ref}.Batch'
                patch_target_forward_ref = ForwardRef(line, patch_target_ref)
                wrapper_name = f'Add{class_name}AttackEffectWrapper'
                wrapper_ref = f'{tech_name}.{wrapper_name}'
                wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects)
                wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
                if isinstance(line, GenieBuildingLineGroup):
                    wrapper_raw_api_object.set_location(f'data/game_entity/generic/{name_lookup_dict[head_unit_id][1]}/')
                    wrapper_raw_api_object.set_filename(f'{tech_lookup_dict[tech_id][1]}_upgrade')
                else:
                    wrapper_raw_api_object.set_location(ForwardRef(tech_group, tech_name))
                nyan_patch_name = f'Add{class_name}AttackEffect'
                nyan_patch_ref = f'{tech_name}.{wrapper_name}.{nyan_patch_name}'
                nyan_patch_location = ForwardRef(tech_group, wrapper_ref)
                nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
                nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
                nyan_patch_raw_api_object.set_patch_target(patch_target_forward_ref)
                attack_ref = f'{nyan_patch_ref}.{class_name}'
                attack_raw_api_object = RawAPIObject(attack_ref, class_name, dataset.nyan_api_objects)
                attack_raw_api_object.add_raw_parent(attack_parent)
                attack_location = ForwardRef(tech_group, nyan_patch_ref)
                attack_raw_api_object.set_location(attack_location)
                type_ref = f'util.attribute_change_type.types.{class_name}'
                change_type = dataset.pregen_nyan_objects[type_ref].get_nyan_object()
                attack_raw_api_object.add_raw_member('type', change_type, effect_parent)
                min_value = dataset.pregen_nyan_objects['effect.discrete.flat_attribute_change.min_damage.AoE2MinChangeAmount'].get_nyan_object()
                attack_raw_api_object.add_raw_member('min_change_value', min_value, effect_parent)
                amount_name = f'{nyan_patch_ref}.{class_name}.ChangeAmount'
                amount_raw_api_object = RawAPIObject(amount_name, 'ChangeAmount', dataset.nyan_api_objects)
                amount_raw_api_object.add_raw_parent('engine.util.attribute.AttributeAmount')
                amount_location = ForwardRef(line, attack_ref)
                amount_raw_api_object.set_location(amount_location)
                attribute = dataset.pregen_nyan_objects['util.attribute.types.Health'].get_nyan_object()
                amount_raw_api_object.add_raw_member('type', attribute, 'engine.util.attribute.AttributeAmount')
                amount_raw_api_object.add_raw_member('amount', attack_amount, 'engine.util.attribute.AttributeAmount')
                line.add_raw_api_object(amount_raw_api_object)
                amount_forward_ref = ForwardRef(line, amount_name)
                attack_raw_api_object.add_raw_member('change_value', amount_forward_ref, effect_parent)
                attack_raw_api_object.add_raw_member('ignore_protection', [], effect_parent)
                line.add_raw_api_object(attack_raw_api_object)
                attack_forward_ref = ForwardRef(line, attack_ref)
                nyan_patch_raw_api_object.add_raw_patch_member('effects', [attack_forward_ref], 'engine.util.effect_batch.EffectBatch', MemberOperator.ADD)
                patch_forward_ref = ForwardRef(tech_group, nyan_patch_ref)
                wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
                tech_group.add_raw_api_object(wrapper_raw_api_object)
                tech_group.add_raw_api_object(nyan_patch_raw_api_object)
                wrapper_forward_ref = ForwardRef(tech_group, wrapper_ref)
                patches.append(wrapper_forward_ref)
            elif isinstance(diff_attack, RightMissingMember):
                attack = diff_attack.ref
                armor_class = attack['type_id'].value
                class_name = armor_lookup_dict[armor_class]
                patch_target_ref = f'{ability_ref}.Batch'
                patch_target_forward_ref = ForwardRef(line, patch_target_ref)
                wrapper_name = f'Remove{class_name}AttackEffectWrapper'
                wrapper_ref = f'{tech_name}.{wrapper_name}'
                wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects)
                wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
                if isinstance(line, GenieBuildingLineGroup):
                    wrapper_raw_api_object.set_location(f'data/game_entity/generic/{name_lookup_dict[head_unit_id][1]}/')
                    wrapper_raw_api_object.set_filename(f'{tech_lookup_dict[tech_id][1]}_upgrade')
                else:
                    wrapper_raw_api_object.set_location(ForwardRef(tech_group, tech_name))
                nyan_patch_name = f'Remove{class_name}AttackEffect'
                nyan_patch_ref = f'{tech_name}.{wrapper_name}.{nyan_patch_name}'
                nyan_patch_location = ForwardRef(tech_group, wrapper_ref)
                nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
                nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
                nyan_patch_raw_api_object.set_patch_target(patch_target_forward_ref)
                attack_ref = f'{ability_ref}.{class_name}'
                attack_forward_ref = ForwardRef(line, attack_ref)
                nyan_patch_raw_api_object.add_raw_patch_member('effects', [attack_forward_ref], 'engine.util.effect_batch.EffectBatch', MemberOperator.SUBTRACT)
                patch_forward_ref = ForwardRef(tech_group, nyan_patch_ref)
                wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
                tech_group.add_raw_api_object(wrapper_raw_api_object)
                tech_group.add_raw_api_object(nyan_patch_raw_api_object)
                wrapper_forward_ref = ForwardRef(tech_group, wrapper_ref)
                patches.append(wrapper_forward_ref)
            else:
                diff_armor_class = diff_attack['type_id']
                if not isinstance(diff_armor_class, NoDiffMember):
                    raise ValueError(f'Could not create effect upgrade for line {repr(line)}: Out of order')
                armor_class = diff_armor_class.ref.value
                attack_amount = diff_attack['amount'].value
                class_name = armor_lookup_dict[armor_class]
                patch_target_ref = f'{ability_ref}.Batch.{class_name}.ChangeAmount'
                patch_target_forward_ref = ForwardRef(line, patch_target_ref)
                wrapper_name = f'Change{class_name}AttackWrapper'
                wrapper_ref = f'{tech_name}.{wrapper_name}'
                wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects)
                wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
                if isinstance(line, GenieBuildingLineGroup):
                    wrapper_raw_api_object.set_location(f'data/game_entity/generic/{name_lookup_dict[head_unit_id][1]}/')
                    wrapper_raw_api_object.set_filename(f'{tech_lookup_dict[tech_id][1]}_upgrade')
                else:
                    wrapper_raw_api_object.set_location(ForwardRef(tech_group, tech_name))
                nyan_patch_name = f'Change{class_name}Attack'
                nyan_patch_ref = f'{tech_name}.{wrapper_name}.{nyan_patch_name}'
                nyan_patch_location = ForwardRef(tech_group, wrapper_ref)
                nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
                nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
                nyan_patch_raw_api_object.set_patch_target(patch_target_forward_ref)
                nyan_patch_raw_api_object.add_raw_patch_member('amount', attack_amount, 'engine.util.attribute.AttributeAmount', MemberOperator.ADD)
                patch_forward_ref = ForwardRef(tech_group, nyan_patch_ref)
                wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
                tech_group.add_raw_api_object(wrapper_raw_api_object)
                tech_group.add_raw_api_object(nyan_patch_raw_api_object)
                wrapper_forward_ref = ForwardRef(tech_group, wrapper_ref)
                patches.append(wrapper_forward_ref)
        return patches

    @staticmethod
    def get_attack_resistances(tech_group: GenieTechEffectBundleGroup, line: GenieGameEntityGroup, diff: ConverterObject, ability_ref: str) -> list[ForwardRef]:
        if False:
            return 10
        '\n        Upgrades resistances that are used for attacking (unit command: 7)\n\n        :param tech_group: Tech that gets the patch.\n        :type tech_group: ...dataformat.converter_object.ConverterObjectGroup\n        :param line: Unit/Building line that gets the ability.\n        :type line: ...dataformat.converter_object.ConverterObjectGroup\n        :param diff: A diff between two ConvertObject instances.\n        :type diff: ...dataformat.converter_object.ConverterObject\n        :param ability_ref: Reference of the ability raw API object the effects are added to.\n        :type ability_ref: str\n        :returns: The forward references for the resistances.\n        :rtype: list\n        '
        head_unit_id = line.get_head_unit_id()
        tech_id = tech_group.get_id()
        dataset = line.data
        patches = []
        name_lookup_dict = internal_name_lookups.get_entity_lookups(dataset.game_version)
        armor_lookup_dict = internal_name_lookups.get_armor_class_lookups(dataset.game_version)
        tech_lookup_dict = internal_name_lookups.get_tech_lookups(dataset.game_version)
        tech_name = tech_lookup_dict[tech_id][0]
        diff_armors = diff['armors'].value
        for diff_armor in diff_armors.values():
            if isinstance(diff_armor, NoDiffMember):
                continue
            if isinstance(diff_armor, LeftMissingMember):
                armor = diff_armor.ref
                armor_class = armor['type_id'].value
                armor_amount = armor['amount'].value
                if armor_class == -1:
                    continue
                class_name = armor_lookup_dict[armor_class]
                resistance_parent = 'engine.resistance.discrete.flat_attribute_change.FlatAttributeChange'
                armor_parent = 'engine.resistance.discrete.flat_attribute_change.type.FlatAttributeChangeDecrease'
                patch_target_ref = f'{ability_ref}'
                patch_target_forward_ref = ForwardRef(line, patch_target_ref)
                wrapper_name = f'Add{class_name}AttackResistanceWrapper'
                wrapper_ref = f'{tech_name}.{wrapper_name}'
                wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects)
                wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
                if isinstance(line, GenieBuildingLineGroup):
                    wrapper_raw_api_object.set_location(f'data/game_entity/generic/{name_lookup_dict[head_unit_id][1]}/')
                    wrapper_raw_api_object.set_filename(f'{tech_lookup_dict[tech_id][1]}_upgrade')
                else:
                    wrapper_raw_api_object.set_location(ForwardRef(tech_group, tech_name))
                nyan_patch_name = f'Add{class_name}AttackResistance'
                nyan_patch_ref = f'{tech_name}.{wrapper_name}.{nyan_patch_name}'
                nyan_patch_location = ForwardRef(tech_group, wrapper_ref)
                nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
                nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
                nyan_patch_raw_api_object.set_patch_target(patch_target_forward_ref)
                attack_ref = f'{nyan_patch_ref}.{class_name}'
                attack_raw_api_object = RawAPIObject(attack_ref, class_name, dataset.nyan_api_objects)
                attack_raw_api_object.add_raw_parent(armor_parent)
                attack_location = ForwardRef(tech_group, nyan_patch_ref)
                attack_raw_api_object.set_location(attack_location)
                type_ref = f'util.attribute_change_type.types.{class_name}'
                change_type = dataset.pregen_nyan_objects[type_ref].get_nyan_object()
                attack_raw_api_object.add_raw_member('type', change_type, resistance_parent)
                amount_name = f'{nyan_patch_ref}.{class_name}.BlockAmount'
                amount_raw_api_object = RawAPIObject(amount_name, 'BlockAmount', dataset.nyan_api_objects)
                amount_raw_api_object.add_raw_parent('engine.util.attribute.AttributeAmount')
                amount_location = ForwardRef(line, attack_ref)
                amount_raw_api_object.set_location(amount_location)
                attribute = dataset.pregen_nyan_objects['util.attribute.types.Health'].get_nyan_object()
                amount_raw_api_object.add_raw_member('type', attribute, 'engine.util.attribute.AttributeAmount')
                amount_raw_api_object.add_raw_member('amount', armor_amount, 'engine.util.attribute.AttributeAmount')
                line.add_raw_api_object(amount_raw_api_object)
                amount_forward_ref = ForwardRef(line, amount_name)
                attack_raw_api_object.add_raw_member('block_value', amount_forward_ref, resistance_parent)
                line.add_raw_api_object(attack_raw_api_object)
                attack_forward_ref = ForwardRef(line, attack_ref)
                nyan_patch_raw_api_object.add_raw_patch_member('resistances', [attack_forward_ref], 'engine.ability.type.Resistance', MemberOperator.ADD)
                patch_forward_ref = ForwardRef(tech_group, nyan_patch_ref)
                wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
                tech_group.add_raw_api_object(wrapper_raw_api_object)
                tech_group.add_raw_api_object(nyan_patch_raw_api_object)
                wrapper_forward_ref = ForwardRef(tech_group, wrapper_ref)
                patches.append(wrapper_forward_ref)
            elif isinstance(diff_armor, RightMissingMember):
                armor = diff_armor.ref
                armor_class = armor['type_id'].value
                class_name = armor_lookup_dict[armor_class]
                patch_target_ref = f'{ability_ref}'
                patch_target_forward_ref = ForwardRef(line, patch_target_ref)
                wrapper_name = f'Remove{class_name}AttackResistanceWrapper'
                wrapper_ref = f'{tech_name}.{wrapper_name}'
                wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects)
                wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
                if isinstance(line, GenieBuildingLineGroup):
                    wrapper_raw_api_object.set_location(f'data/game_entity/generic/{name_lookup_dict[head_unit_id][1]}/')
                    wrapper_raw_api_object.set_filename(f'{tech_lookup_dict[tech_id][1]}_upgrade')
                else:
                    wrapper_raw_api_object.set_location(ForwardRef(tech_group, tech_name))
                nyan_patch_name = f'Remove{class_name}AttackResistance'
                nyan_patch_ref = f'{tech_name}.{wrapper_name}.{nyan_patch_name}'
                nyan_patch_location = ForwardRef(tech_group, wrapper_ref)
                nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
                nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
                nyan_patch_raw_api_object.set_patch_target(patch_target_forward_ref)
                attack_ref = f'{ability_ref}.{class_name}'
                attack_forward_ref = ForwardRef(line, attack_ref)
                nyan_patch_raw_api_object.add_raw_patch_member('resistances', [attack_forward_ref], 'engine.ability.type.Resistance', MemberOperator.SUBTRACT)
                patch_forward_ref = ForwardRef(tech_group, nyan_patch_ref)
                wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
                tech_group.add_raw_api_object(wrapper_raw_api_object)
                tech_group.add_raw_api_object(nyan_patch_raw_api_object)
                wrapper_forward_ref = ForwardRef(tech_group, wrapper_ref)
                patches.append(wrapper_forward_ref)
            else:
                diff_armor_class = diff_armor['type_id']
                if not isinstance(diff_armor_class, NoDiffMember):
                    raise ValueError(f'Could not create effect upgrade for line {repr(line)}: Out of order')
                armor_class = diff_armor_class.ref.value
                armor_amount = diff_armor['amount'].value
                class_name = armor_lookup_dict[armor_class]
                patch_target_ref = f'{ability_ref}.{class_name}.BlockAmount'
                patch_target_forward_ref = ForwardRef(line, patch_target_ref)
                wrapper_name = f'Change{class_name}ResistanceWrapper'
                wrapper_ref = f'{tech_name}.{wrapper_name}'
                wrapper_raw_api_object = RawAPIObject(wrapper_ref, wrapper_name, dataset.nyan_api_objects)
                wrapper_raw_api_object.add_raw_parent('engine.util.patch.Patch')
                if isinstance(line, GenieBuildingLineGroup):
                    wrapper_raw_api_object.set_location(f'data/game_entity/generic/{name_lookup_dict[head_unit_id][1]}/')
                    wrapper_raw_api_object.set_filename(f'{tech_lookup_dict[tech_id][1]}_upgrade')
                else:
                    wrapper_raw_api_object.set_location(ForwardRef(tech_group, tech_name))
                nyan_patch_name = f'Change{class_name}Resistance'
                nyan_patch_ref = f'{tech_name}.{wrapper_name}.{nyan_patch_name}'
                nyan_patch_location = ForwardRef(tech_group, wrapper_ref)
                nyan_patch_raw_api_object = RawAPIObject(nyan_patch_ref, nyan_patch_name, dataset.nyan_api_objects, nyan_patch_location)
                nyan_patch_raw_api_object.add_raw_parent('engine.util.patch.NyanPatch')
                nyan_patch_raw_api_object.set_patch_target(patch_target_forward_ref)
                nyan_patch_raw_api_object.add_raw_patch_member('amount', armor_amount, 'engine.util.attribute.AttributeAmount', MemberOperator.ADD)
                patch_forward_ref = ForwardRef(tech_group, nyan_patch_ref)
                wrapper_raw_api_object.add_raw_member('patch', patch_forward_ref, 'engine.util.patch.Patch')
                tech_group.add_raw_api_object(wrapper_raw_api_object)
                tech_group.add_raw_api_object(nyan_patch_raw_api_object)
                wrapper_forward_ref = ForwardRef(tech_group, wrapper_ref)
                patches.append(wrapper_forward_ref)
        return patches